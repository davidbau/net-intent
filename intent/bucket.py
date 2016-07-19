"""Convolutional network example.

Run the training for 20 epochs with
```
python run.py --num-epochs 20
```

"""
import logging
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer

from theano import tensor

from blocks.bricks.conv import Convolutional
from blocks.bricks import application
from blocks.bricks import Brick
from blocks.bricks import Linear
from blocks.bricks import Rectifier
from blocks.bricks import Activation
from blocks.bricks import Softmax
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.cost import MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.filter import get_brick
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.monitoring.aggregation import AggregationScheme
from blocks.monitoring.aggregation import Aggregator
from blocks.serialization import load_parameters
from blocks.utils import dict_subset
from blocks.utils import shared_floatx
from blocks.utils import shared_like
from collections import OrderedDict
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from intent.ablation import ConfusionMatrix
from intent.ablation import ConfusionImage
from intent.ablation import Sum
from intent.ablation import ablate_inputs
from intent.lenet import create_sorted_lenet
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import make_mask
from intent.rf import layerarray_fieldmap
from prior import create_fair_basis
from theano import gradient
from theano import tensor
from theano.ifelse import ifelse
from theano.printing import Print
import theano
import numpy
import numbers
import pickle

# For testing
from blocks.roles import OUTPUT, BIAS

class Concatenate(AggregationScheme):
    """Aggregation scheme which concatenates results on axis 0.
    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        Theano variable for the quantity to be concatenated
    """
    def __init__(self, variable):
        self.mnist_test = MNIST(("test",))
        self.variable = variable

    def get_aggregator(self):
        initialized = shared_like(0.)
        total_acc = shared_like(self.variable)
        empty_init = tensor.zeros(
                (0,) * self.variable.ndim, dtype=self.variable.dtype)

        empty_shape = tuple([
            0 if d is 0
            else self.variable.shape[d]
            for d in range(self.variable.ndim)])
        conditional_update_num = tensor.concatenate([
            ifelse(initialized, total_acc, empty_init.reshape(empty_shape)),
            self.variable])

        initialization_updates = [(total_acc, empty_init),
                                  (initialized, tensor.zeros_like(initialized))]

        accumulation_updates = [(total_acc, conditional_update_num),
                                (initialized, tensor.ones_like(initialized))]

        aggregator = Aggregator(aggregation_scheme=self,
                                initialization_updates=initialization_updates,
                                accumulation_updates=accumulation_updates,
                                readout_variable=(total_acc))

        return aggregator

class MaxActivationTable(Brick):
    """Returns the maximum activation of the given unit for each instance.
    """
    @application(outputs=["max_activation_table"])
    def apply(self, outs):
        all_units = []
        for o in outs:
            while o.ndim > 2:
                o = o.max(axis=o.ndim - 1)
            all_units.append(o)
        return tensor.concatenate(all_units, axis=1)
        # return all_units[1]

class BucketVisualizer:
    def __init__(self, save_to, act_table):
        self.mnist_test = MNIST(("test",), sources=['features', 'targets'])
        self.table = self.load_act_table(save_to, act_table)

    def all_match(self, index, the_set, positive):
        if the_set is None or len(the_set) == 0:
            return True
        selected = self.table[index, the_set]
        if positive:
            matched = selected > 0
        else:
            matched = selected <= 0
        return matched.sum() == len(the_set)

    def activations_for_sample(self, index):
        return self.table[index, :]

    def positive_for_sample(self, index):
        return numpy.where(self.activations_for_sample(index) > 0)[0]

    def negative_for_sample(self, index):
        return numpy.where(self.activations_for_sample(index) <= 0)[0]

    def prediction_for_sample(self, index):
        return self.table[index, :10].argmax()

    def label_for_sample(self, index):
        return self.mnist_test.get_data(request=index)[1][0]

    def filter_image_bytes(self,
            positive_set=None, negative_set=None, sort_by=None,
            columns=100, limit=None, ulimit=None, descending=False):
        include_indexes = [ind for ind in range(self.table.shape[0])
                if (self.all_match(ind, positive_set, True) and
                    self.all_match(ind, negative_set, False))]
        if sort_by:
            include_indexes.sort(key=lambda x: self.table[x, sort_by].sum())
        if descending:
            include_indexes.reverse()
        if limit or ulimit and not(
                limit and ulimit and limit + ulimit >= len(include_indexes)):
            lower = include_indexes[:limit] if limit else []
            upper = include_indexes[-ulimit:] if ulimit else []
            include_indexes = lower + upper
        count = max(1, len(include_indexes))
        grid_shape = (((count - 1) // columns + 1), min(columns, count))

        filmstrip = Filmstrip(image_shape=(28, 28), grid_shape=grid_shape)
        for i, index in enumerate(include_indexes):
            filmstrip.set_image((i // columns, i % columns),
                        self.mnist_test.get_data(request=index)[0])
        return filmstrip.save_bytes()

    def example_count(self):
        return self.table.shape[0]

    def unit_count(self):
        return self.table.shape[1]

    def load_act_table(self, save_to, act_table):
        try:
            return pickle.load(open(act_table, 'rb'))
        except FileNotFoundError:
            return self.create_act_table(save_to, act_table)

    def create_act_table(self, save_to, act_table):
        batch_size = 500
        image_size = (28, 28)
        output_size = 10
        convnet = create_sorted_lenet()
        layers = convnet.layers

        x = tensor.tensor4('features')
        y = tensor.lmatrix('targets')

        # Normalize input and apply the convnet
        probs = convnet.apply(x)
        cg = ComputationGraph([probs])

        def full_brick_name(brick):
            return '/'.join([''] + [b.name for b in brick.get_unique_path()])

        # Find layer outputs to probe
        outmap = OrderedDict((full_brick_name(get_brick(out)), out)
                for out in VariableFilter(
                    roles=[OUTPUT], bricks=[Convolutional, Linear])(
                        cg.variables))
        # Generate pics for biases
        biases = VariableFilter(roles=[BIAS])(cg.parameters)

        # Generate parallel array, in the same order, for outputs
        outs = [outmap[full_brick_name(get_brick(b))] for b in biases]

        # Figure work count
        error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                      .copy(name='error_rate'))
        max_activation_table = (MaxActivationTable().apply(
                outs).copy(name='max_activation_table'))
        max_activation_table.tag.aggregation_scheme = (
                Concatenate(max_activation_table))

        model = Model([
            error_rate,
            max_activation_table])

        # Load it with trained parameters
        params = load_parameters(open(save_to, 'rb'))
        model.set_parameter_values(params)

        mnist_test_stream = DataStream.default_stream(
            self.mnist_test,
            iteration_scheme=SequentialScheme(
                self.mnist_test.num_examples, batch_size))

        evaluator = DatasetEvaluator([
            error_rate,
            max_activation_table
            ])
        results = evaluator.evaluate(mnist_test_stream)
        table = results['max_activation_table']
        pickle.dump(table, open(act_table, 'wb'))
        return table

class QueryHTTPServer(HTTPServer):
    def __init__(self, tester, *args, **kw):
        super(QueryHTTPServer, self).__init__(*args, **kw)
        self.tester = tester

# HTTPRequestHandler class
class QueryRequestHandler(BaseHTTPRequestHandler):
 
    # GET
    def do_GET(self):
        from urllib.parse import urlparse, parse_qsl
        url = urlparse(self.path)
        fields = OrderedDict(parse_qsl(url.query))
        self.dispatch(url, fields)

    def do_POST(self):
        from urllib.parse import urlparse, parse_qsl
        url = urlparse(self.path)
        fields = OrderedDict(parse_qsl(url.query))
        length = int(self.headers.getheader('content-length'))
        field_data = self.rfile.read(length)
        fields.update(urlparse.parse_qsl(field_data))
        self.dispatch(url, fields)

    def dispatch(self, url, fields):
        if url.path == '/bucket':
            self.bucket(url, fields)
            return
        if url.path == '/similar_to':
            self.similar_to(url, fields)
            return
        if url.path == '/examples':
            self.examples(url, fields)
            return
        if url.path == '/units':
            self.units(url, fields)
            return

        # Send response status code
        self.send_response(200)
 
        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()
 
        # Send message back to client
        message = '<pre>No handler for<br>' + repr(url) + '<br>' + repr(fields)
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))

    def examples(self, url, fields):
        from urllib.parse import urlencode
        html = []
        num = 30
        if 'num' in fields:
            num = int(fields['num'])
        for x in range(num):
            html.append(
                '<nobr>%d:%d-%d<img src="/similar_to?example=%d&%s"></nobr><br>' % (
                    x,
                    self.server.tester.label_for_sample(x),
                    self.server.tester.prediction_for_sample(x),
                    x,
                    urlencode(fields)))
        # Form HTML page response
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write(bytes('\n'.join(html), "utf8"))

    def units(self, url, fields):
        from urllib.parse import urlencode
        html = []
        limit = ulimit = 20
        if 'limit' in fields:
            limit = int(fields['limit'])
        if 'ulimit' in fields:
            ulimit = int(fields['ulimit'])
        for u in range(self.server.tester.unit_count()):
            html.append(
                '<nobr>%d<img src="/bucket?sort_by=%d&limit=%d&ulimit=%d"></nobr><br>'
                % (u, u, limit, ulimit))
        # Form HTML page response
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write(bytes('\n'.join(html), "utf8"))

    def similar_to(self, url, fields):
        from urllib.parse import urlencode
        example = None
        if 'example' in fields:
            example = int(fields['example'])
        descending = False
        signs = 'PN'
        if 'signs' in fields:
            signs = fields['signs']
        rangepair = None
        if 'range' in fields:
            rangepair = [int(u) for u in fields['range'].split(',')]
        query = []
        if 'P' in signs:
            inds = self.server.tester.positive_for_sample(example)
            if rangepair:
                inds = [i for i in inds if i >= rangepair[0] and i < rangepair[1]]
            query.append('positive=' + ','.join([str(u) for u in inds]))
        if 'N' in signs:
            if len(query): query += '&'
            inds = self.server.tester.negative_for_sample(example)
            if rangepair:
                inds = [i for i in inds if i >= rangepair[0] and i < rangepair[1]]
            query.append('negative=' + ','.join([str(u) for u in inds]))
        forward_keys = dict((k, v) for k, v in fields.items()
                if k not in ['example', 'signs', 'range'])
        if len(forward_keys):
            query.append(urlencode(forward_keys))

        self.send_response(302)
        self.send_header('Location', '/bucket?%s' % '&'.join(query))
        self.end_headers()

    def bucket(self, url, fields):
        positive = []
        if 'positive' in fields:
            positive = [int(u) for u in fields['positive'].split(',')]
        negative = []
        if 'negative' in fields:
            negative = [int(u) for u in fields['negative'].split(',')]
        sort_by = []
        if 'sort_by' in fields:
            sort_by = [int(u) for u in fields['sort_by'].split(',')]
        columns = 100
        if 'columns' in fields:
            columns = int(fields['columns'])
        limit = None
        if 'limit' in fields:
            limit = int(fields['limit'])
        ulimit = None
        if 'ulimit' in fields:
            ulimit = int(fields['ulimit'])
        descending = False
        if 'descending' in fields:
            descending = True
        result = self.server.tester.filter_image_bytes(
                positive, negative, sort_by, columns, limit, ulimit, descending)
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        self.wfile.write(result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Gradient descent vis for the MNIST dataset.")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("act_table", default="activations.pkl", nargs="?",
                        help="Destination to save hidden activations.")
    args = parser.parse_args()
    visualizer = BucketVisualizer(**vars(args))
    port = 8000
    server_address = ('127.0.0.1', port)
    httpd = QueryHTTPServer(visualizer, server_address, QueryRequestHandler)
    print('running server on port %d' % port)
    httpd.serve_forever()
