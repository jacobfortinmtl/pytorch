import os
from nan_ops import NaNConv2d, NaNPool2d, count_skip_conv2d

# Extract and process environment variables early
nan_active = os.environ.get('NAN_ACTIVE', 'false') == "true"
nanconv_threshold = float(os.environ.get('NANCONV_THRESHOLD', '0.0'))
padding = int((self.params["kernel_h"] - 1) / 2)

# Use NaNConv2d if NAN_ACTIVE is true, otherwise use the standard convolution
if nan_active:
    conv0 = NaNConv2d(
        train=False, 
        kernel=self.conv0.weight, 
        bias=self.conv0.bias, 
        padding=padding, 
        stride=self.params["stride_conv"], 
        threshold=nanconv_threshold
    )
    x0 = conv0(x0_bn)
else:
    x0 = self.conv0(x0_bn)

# Print SKIP count
skip_count = count_skip_conv2d(
    x0_bn, 
    self.conv0.weight, 
    padding=padding, 
    stride=self.params["stride_conv"], 
    threshold=nanconv_threshold
)
print('SKIP count', skip_count)