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
from blocks.bricks import Linear
from blocks.bricks import Rectifier
from blocks.bricks import Activation
from blocks.bricks import Softmax
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
from blocks.serialization import load_parameters
from blocks.utils import shared_floatx
from blocks.utils import dict_subset
from collections import OrderedDict
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from intent.ablation import ConfusionMatrix
from intent.ablation import ConfusionImage
from intent.ablation import Sum
from intent.ablation import ablate_inputs
from intent.lenet import create_lenet_5
from intent.maxact import MaximumActivationSearch
from intent.filmstrip import Filmstrip
from intent.rf import make_mask
from intent.rf import layerarray_fieldmap
from prior import create_fair_basis
from theano import gradient
from theano import tensor
import theano
import numpy
import numbers

# For testing
from blocks.roles import OUTPUT

def extract_sample(activations, data_stream, n=2000):
    cg = ComputationGraph(activations)
    input_names = [v.name for v in cg.inputs]
    fn = theano.function(cg.inputs, [activations])
    result = None
    for batch in data_stream.get_epoch_iterator(as_dict=True):
        values = fn(**dict_subset(batch, input_names))
        if result is None:
            result = values[0]
        else:
            result = numpy.concatenate((result, values[0]))
        if result.shape[0] >= n:
            result = result[(slice(0, n), ) +
                    (slice(None),) * (len(result.shape) - 1)]
            return result

class AblationTester:
    def __init__(self, save_to):
        batch_size = 500
        image_size = (28, 28)
        output_size = 10
        convnet = create_lenet_5()
        layers = convnet.layers

        logging.info("Input dim: {} {} {}".format(
            *convnet.children[0].get_dim('input_')))
        for i, layer in enumerate(convnet.layers):
            if isinstance(layer, Activation):
                logging.info("Layer {} ({})".format(
                    i, layer.__class__.__name__))
            else:
                logging.info("Layer {} ({}) dim: {} {} {}".format(
                    i, layer.__class__.__name__, *layer.get_dim('output')))

        mnist_test = MNIST(("test",), sources=['features', 'targets'])
        basis = create_fair_basis(mnist_test, 10, 10)

        x = tensor.tensor4('features')
        y = tensor.lmatrix('targets')

        # Normalize input and apply the convnet
        probs = convnet.apply(x)
        cg = ComputationGraph([probs])

        def full_brick_name(brick):
            return '/'.join([''] + [b.name for b in brick.get_unique_path()])

        # Find layer outputs to probe
        outs = OrderedDict((full_brick_name(get_brick(out)), out)
                for out in VariableFilter(
                    roles=[OUTPUT], bricks=[Convolutional, Linear])(
                        cg.variables))

        # Normalize input and apply the convnet
        error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                      .copy(name='error_rate'))
        confusion = (ConfusionMatrix().apply(y.flatten(), probs)
                      .copy(name='confusion'))
        confusion.tag.aggregation_scheme = Sum(confusion)
        confusion_image = (ConfusionImage().apply(y.flatten(), probs, x)
                      .copy(name='confusion_image'))
        confusion_image.tag.aggregation_scheme = Sum(confusion_image)

        model = Model(
                [error_rate, confusion, confusion_image] + list(outs.values()))

        # Load it with trained parameters
        params = load_parameters(open(save_to, 'rb'))
        model.set_parameter_values(params)

        mnist_test = MNIST(("test",))
        mnist_test_stream = DataStream.default_stream(
            mnist_test,
            iteration_scheme=SequentialScheme(
                mnist_test.num_examples, batch_size))

        self.model = model
        self.mnist_test_stream = mnist_test_stream
        self.evaluator = DatasetEvaluator(
                [error_rate, confusion, confusion_image])
        self.base_results = self.evaluator.evaluate(mnist_test_stream)

        # TODO: allow target layer to be parameterized
        self.target_layer = '/lenet/mlp/linear_0'
        self.next_layer_param = '/lenet/mlp/linear_1.W'
        self.base_sample = extract_sample(
                outs[self.target_layer], mnist_test_stream)
        self.base_param_value = (
            model.get_parameter_dict()[
                self.next_layer_param].get_value().copy())


    def probe_ablation(self, targets, differential=True, compensate=False):
        # Probe the given layer

        # Figure neurons to ablate
        # hist = histograms[('linear_1', 'b')]
        # targets = [57]

        # Now adjust the next layer weights based on the probe
        param = self.model.get_parameter_dict()[self.next_layer_param]
        if len(targets):
            new_weights = ablate_inputs(
                targets,
                self.base_sample,
                param.get_value(),
                compensate=compensate)
            param.set_value(new_weights)
        # Evaluation pass
        result = self.evaluator.evaluate(self.mnist_test_stream)
        # Reset params back to baseline value
        param.set_value(self.base_param_value)
        # Result contains error_rate, confusion, and confusion_image
        if differential:
            for key in result:
                result[key] -= self.base_results[key]
        return result

class QueryHTTPServer(HTTPServer):
    def __init__(self, tester, *args, **kw):
        super(QueryHTTPServer, self).__init__(*args, **kw)
        self.tester = tester

# HTTPRequestHandler class
class QueryRequestHandler(BaseHTTPRequestHandler):
 
    # GET
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        url = urlparse(self.path)
        fields = parse_qs(url.query)
        self.dispatch(url, fields)

    def do_POST(self):
        from urllib.parse import urlparse, parse_qs
        url = urlparse(self.path)
        fields = parse_qs(url.query)
        length = int(self.headers.getheader('content-length'))
        field_data = self.rfile.read(length)
        fields.update(urlparse.parse_qs(field_data))
        self.dispatch(url, fields)

    def dispatch(self, url, fields):
        if url.path == '/ablate':
            self.ablate(url, fields)
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

    def ablate(self, url, fields):
        units = []
        if 'units' in fields:
           units = [int(u) for u in fields['units'][0].split(',')]
        compensate = False
        if 'compensate' in fields:
            compensate = True
        result = self.server.tester.probe_ablation(units, compensate=compensate)
        confusion_image = result['confusion_image']
        filmstrip = Filmstrip(image_shape=confusion_image.shape[-2:],
                grid_shape=confusion_image.shape[:2])
        for goal in range(confusion_image.shape[0]):
            for actual in range(confusion_image.shape[1]):
                sum_image = confusion_image[goal, actual, :, :, :]
                filmstrip.set_image(
                        (goal, actual),
                        (sum_image - sum_image.min()) /
                        (sum_image.max() - sum_image.min() + 1e-9))
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        self.wfile.write(filmstrip.save_bytes())
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Gradient descent vis for the MNIST dataset.")
    parser.add_argument("save_to", default="mnist.tar", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    args = parser.parse_args()
    tester = AblationTester(**vars(args))
    port = 8000
    server_address = ('127.0.0.1', port)
    httpd = QueryHTTPServer(tester, server_address, QueryRequestHandler)
    print('running server on port %d' % port)
    httpd.serve_forever()
