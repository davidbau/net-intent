# Smaller movie:
# -vf "scale=480:480,pad=640:480:80:0"

KIND=syn

mkdir -p movies/$KIND


avconv -y -framerate 30 -f image2 -qscale 1 \
        -i pics/$KIND/composite_%04d.jpg \
        -c:v h264 -preset veryslow -tune grain -crf 23 \
        -vf "scale=1739:1080,pad=1920:1080:90:0" \
        movies/$KIND/composite.mov

# avconv -y -framerate 30 -f image2 -qscale 1 \
#         -i pics/$KIND/linear_2_b_%04d.jpg \
#         -vf "scale=1080:1080,pad=1920:1080:420:0" \
#         movies/$KIND/linear_2.mov

# avconv -y -framerate 30 -f image2 -qscale 1 \
#         -i pics/$KIND/linear_1_b_%04d.jpg \
#         -vf "scale=1157:1080,pad=1920:1080:381:0" \
#         movies/$KIND/linear_1.mov

# avconv -y -framerate 30 -f image2 -qscale 1 \
#         -i pics/$KIND/linear_0_b_%04d.jpg \
#         -vf "scale=1440:1080,pad=1920:1080:240:0" \
#         movies/$KIND/linear_0.mov

# avconv -y -framerate 30 -f image2 -qscale 1 \
#         -i pics/$KIND/conv_1_b_%04d.jpg \
#         -vf "scale=674:1080,pad=1920:1080:623:0" \
#         movies/$KIND/conv_1.mov

# avconv -y -framerate 30 -f image2 -qscale 1 \
#         -i pics/$KIND/conv_0_b_%04d.jpg \
#         -vf "scale=1804:1080,pad=1920:1080:58:0" \
#         movies/$KIND/conv_0.mov
