import torch
import torch.nn.functional as F
import numpy as np
import unittest
import os
import math

class TestNaNPool2dPy(unittest.TestCase):
    

    def unit_test_indices(self, input_tensor, expected, name, test_torch=True, test_expected=True):
        from nan_ops import NaNPool2d

        
        nanPoolPy = NaNPool2d(max_threshold=1) 
        torchpool = torch.nn.MaxPool2d(2, 2, return_indices=True)

        default = torchpool(input_tensor)[1]
        testing = nanPoolPy(input_array=input_tensor, pool_size=(2, 2), strides=(2, 2))[1]

        if test_torch:
            self.assertTrue(torch.equal(default, testing), f"{name} Test failed: Torch and NaNPool indices do not match.")
            print(f"{name} Test passed: Torch and NaNPool indices match.")
        else: print('NaNPool expected behaviour differs from Torch -- comparison to Torch is skipped')
        # print(default , testing)

        if test_expected:
            self.assertTrue(torch.equal(expected, testing), f"{name} Test failed: Expected and NaNPool indices do not match.")
            print(f"{name} Test passed: Expected and NaNPool indices match.")
        else: print('NanPool expected behaviour matches Torch -- no need to run additional test')
        # print(expected , testing)

    def unit_test_maxvalues(self, input_tensor, expected, name, test_torch=True, test_expected=True):
        from nan_ops import NaNPool2d

        nanPoolPy = NaNPool2d(max_threshold=1) 
        torchpool = torch.nn.MaxPool2d(2, 2, return_indices=True)

        default = torchpool(input_tensor)[0]
        default = default.masked_fill(torch.isnan(default), 0.5)
        testing = nanPoolPy(input_array=input_tensor, pool_size=(2, 2), strides=(2, 2))[0]
        testing = testing.masked_fill(torch.isnan(testing), 0.5)
        # print(default , testing)

        if test_torch:
            self.assertTrue(torch.equal(default, testing), f"{name} Test failed: Torch and NaNPool max values do not match.")
            print(f"{name} Test passed: Torch and NaNPool max values match.")
        else: print('NaNPool expected behaviour differs from Torch -- comparison to Torch is skipped')
        # print(default , testing)

        if test_expected:
            self.assertTrue(torch.equal(expected, testing), f"{name} Test failed: Expected and NaNPool max values do not match.")
            print(f"{name} Test passed: Expected and NaNPool max values match.")
        else: print('NanPool expected behaviour matches Torch -- no need to run additional test')
        # print(expected , testing)

    def test_all_nans(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]]]
        """
        input = torch.randint(0, 100, (2, 2, 2, 2)).float()
        input[:] = float('nan')
        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[0]], [[0]]], [[[0]], [[0]]]])
        expected_val = torch.tensor([[[[0.5]], [[0.5]]], [[[0.5]], [[0.5]]]])
        self.unit_test_indices(input, expected_idx, name='NaNPool all nans', test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, name='NaNPool all nans', test_torch=True, test_expected=True)

    def test_no_nans(self):
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

         [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

         [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        shape = (2, 2, 2, 4)
        num_elements = np.prod(shape)
        # Generate unique random values
        unique_values = torch.randperm(num_elements) % 100  # Ensure values are within the range 0 to 99
        # Reshape to desired shape and convert to float
        input = unique_values.reshape(shape).float()
        # print('INPUT:\n', input, input.shape)
        self.unit_test_indices(input, None, name='NaNPool no nans', test_torch=True, test_expected=False)
        self.unit_test_maxvalues(input, None, name='NaNPool no nans', test_torch=True, test_expected=False)

    def test_mixed_nans_multimaxval(self):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

         [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, 10.]]]]
        """
        input = torch.randint(0, 100, (2, 2, 2, 2)).float()
        input[:] = float('nan')
        input[0] = 1
        input[0, 0, 0, 0] = float('nan')
        input[1,1,1,1] = 10
        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[0]], [[0]]], [[[0]], [[3]]]])
        expected_val = torch.tensor([[[[0.5]], [[0.5]]], [[[0.5]], [[10]]]])
        self.unit_test_indices(input, expected_idx, name='NaNPool mixed nans multi max', test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, name='NaNPool mixed nans multi max', test_torch=False, test_expected=True)

    def test_mixed_nans_no_multimaxval(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [33.,  3.]],

         [[75., 59.],
          [53., 24.]]],


        [[[61., 64.],
          [26., 27.]],

         [[67., 68.],
          [11., nan]]]]
        """
        input = torch.tensor([[[[float('nan'), float('nan')], [33.,  3.]],
                         [[75., 59.], [53., 24.]]],
                         [[[61., 64.], [26., 27.]],
                          [[67., 68.], [11., float('nan')]]]])

        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[2]], [[0]]], [[[1]], [[1]]]])
        expected_val = torch.tensor([[[[33]], [[75]]], [[[64]], [[68]]]])
        self.unit_test_indices(input, expected_idx, name='NaNPool mixed nans no multi max', test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, name='NaNPool mixed nans no multi max', test_torch=False, test_expected=True)


class TestNaNConv2dPy(unittest.TestCase):
    
    def setUp(self):
        # Default values
        self.nonan_fault_tolerance = getattr(self, 'nonan_fault_tolerance', 0.0005)
        self.mixed_fault_tolerance = getattr(self, 'mixed_fault_tolerance', 0.0005)

    def test_all_nans(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]]]
        """
        from nan_ops import NaNConv2d

        
        kernel = torch.randn(2, 2, 3, 3)
        inputs = torch.randn(1, 2, 5, 5)
        inputs[:] = float('nan')

        output = torch.zeros((1,2,5,5))
        output[:,:, 1:-1, 1:-1] = float('nan')
        mask = torch.isnan(output)
        output[mask] = 1

        conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
        nanoutput = conv(inputs)
        nanoutput[mask] = 1
        
        self.assertTrue(torch.equal(output, nanoutput), "NanConv all nans Test failed: Torch and NaNConv values do not match.")
        print("NanConv all nans Test passed: Torch and NaNConv values match.")


    def test_no_nans(self): 
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

         [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

         [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        from nan_ops import NaNConv2d

        fail=0
        n = 1

        for _ in range(n):
          kernel = torch.randn(2, 2, 3, 3)
          inputs = torch.randn(1, 2, 5, 5)

          output = F.conv2d(inputs, kernel, stride=1, padding=1)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-09).all(): fail += 1

        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        fail=0
        for _ in range(n):
          kernel = torch.randn(2, 2, 3, 3)
          inputs = torch.randn(1, 2, 5, 5)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=0, stride=1, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-09).all(): fail += 1

        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=0 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=0 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        fail=0
        for _ in range(n):
          kernel = torch.randn(2, 2, 3, 3)
          inputs = torch.randn(1, 2, 5, 5)

          output = F.conv2d(inputs, kernel, stride=3, padding=1)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=3, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-09).all(): fail += 1

        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=3 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=3 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")


    def test_mixed_nans(self):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

         [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, 10.]]]]
        """
        from nan_ops import NaNConv2d

   
        fail = 0
        n = 1
        for _ in range(n):
            kernel = torch.randn(2, 2, 3, 3)
            inputs = torch.randn(1, 2, 5, 5)

            inputs[:,:, ::2, 0] = float('nan')
            inputs[:,:, ::2, -1] = float('nan')
            # inputs[:,:, -1, :] = float('nan')
            inputs[:,:, 2, :] = float('nan')

            output = F.conv2d(inputs, kernel, stride=1, padding=1)

            conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
            nanoutput = conv(inputs)

            mask = torch.isnan(output)
            output[mask] = 1

            nanoutput[mask] = 1
            if not torch.isclose(nanoutput, output, rtol=1e-10).all(): fail += 1
    
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv mixed nan Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=3 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv mixed nan Test passed: Torch and NaNConv convolutions match for padding=1 and stride=3 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


class PrepNaNConv2d:
    
    def __init__(self, save_dir='./'):
       self.save_dir = save_dir


    def prep_multichannel(self, multichannel_filename='custom'):
        
        torch.manual_seed(0)

        channels = 2
        batches = 2
        size = 4**2
        sqrt_size = int(math.sqrt(size))
        # Determine how many elements should be NaN
        num_nans = int(size * 0.33)
        n = 1


        inputs = torch.arange(1, size*channels*batches + 1).float()

        # Reshape the tensor to the desired shape
        inputs = inputs.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

        # Create a random weights tensor of integers
        kernel = torch.ones(1, 1*channels, 2, 2).float()
        
        total_inputs = []
        # total_kernel = []
        total_output = []

        for _ in range(n):
          
          # Create a random permutation of the indices
          indices = torch.randperm(size)
          
          # Set the elements at the indices to NaN for each channel
          for c in range(inputs.shape[1]):
              inputs[0, c].view(-1)[indices[:num_nans]] = float('nan')

          # total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        #total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, './nanconv_unittests/inputs_multichannel.pt')
        # torch.save(total_kernel, './nanconv_unittests/kernel_multichannel.pt')
        torch.save(total_output, f'./nanconv_unittests/conv_{multichannel_filename}_multichannel.pt')



    def prep_multibatch(self, multibatch_filename='custom'):
        
        torch.manual_seed(0)

        channels = 1
        batches = 5
        size = 4**2
        sqrt_size = int(math.sqrt(size))
        # Determine how many elements should be NaN
        num_nans = int(size * 0.33)
        n = 1


        inputs = torch.arange(1, size*channels*batches + 1).float()

        # Reshape the tensor to the desired shape
        inputs = inputs.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

        # Create a random weights tensor of integers
        kernel = torch.ones(1, 1*channels, 2, 2).float()
        
        total_inputs = []
        # total_kernel = []
        total_output = []

        for _ in range(n):
          
          # Set the elements at the indices to NaN for each channel
          for c in range(inputs.shape[0]):
              # Create a random permutation of the indices
              indices = torch.randperm(size)

              inputs[c, 0].view(-1)[indices[:num_nans]] = float('nan')

          # total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        #total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, './nanconv_unittests/inputs_multibatch.pt')
        # torch.save(total_kernel, './nanconv_unittests/kernel_multibatch.pt')
        torch.save(total_output, f'./nanconv_unittests/conv_{multibatch_filename}_multibatch.pt')


    def prep_multi4d(self, multi4d_filename='custom'):
        
        torch.manual_seed(0)

        channels = 3
        batches = 3
        size = 4**2
        sqrt_size = int(math.sqrt(size))
        # Determine how many elements should be NaN
        num_nans = int(size * 0.33)
        n = 1


        inputs = torch.arange(1, size*channels*batches + 1).float()

        # Reshape the tensor to the desired shape
        inputs = inputs.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

        # Create a random weights tensor of integers
        kernel = torch.ones(1, 1*channels, 2, 2).float()
        
        total_inputs = []
        # total_kernel = []
        total_output = []

        for _ in range(n):
          
          # Set the elements at the indices to NaN for each channel
          # Set the elements at the indices to NaN for each channel
          for b in range(inputs.shape[0]):
              for c in range(inputs.shape[1]):
                  # Create a random permutation of the indices
                  indices = torch.randperm(size)

                  inputs[b, c].view(-1)[indices[:num_nans]] = float('nan')

          # total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        #total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, './nanconv_unittests/inputs_multi4d.pt')
        # torch.save(total_kernel, './nanconv_unittests/kernel_multi4d.pt')
        torch.save(total_output, f'./nanconv_unittests/conv_{multi4d_filename}_multi4d.pt')
        
    
    def prep_no_nans(self, nonan_filename='custom'): 
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

         [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

         [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        torch.manual_seed(0)

        n = 1
        
        total_inputs = []
        total_kernel = []
        total_output = []

        for _ in range(n):
          kernel = torch.randn(2, 2, 3, 3)
          inputs = torch.randn(1, 2, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=1)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, './nanconv_unittests/inputs_stride1_pad1_nonans.pt')
        torch.save(total_kernel, './nanconv_unittests/kernel_stride1_pad1_nonans.pt')
        torch.save(total_output, f'./nanconv_unittests/conv_{nonan_filename}_stride1_pad1_nonans.pt')


        total_inputs = []
        total_kernel = []
        total_output = []
        for _ in range(n):
          kernel = torch.randn(2, 2, 3, 3)
          inputs = torch.randn(1, 2, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)
          
          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, './nanconv_unittests/inputs_stride1_pad0_nonans.pt')
        torch.save(total_kernel, './nanconv_unittests/kernel_stride1_pad0_nonans.pt')
        torch.save(total_output, f'./nanconv_unittests/conv_{nonan_filename}_stride1_pad0_nonans.pt')

        
        total_inputs = []
        total_kernel = []
        total_output = []
        for _ in range(n):
          kernel = torch.randn(2, 2, 3, 3)
          inputs = torch.randn(1, 2, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)
          
          output = F.conv2d(inputs, kernel, stride=3, padding=1)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, './nanconv_unittests/inputs_stride3_pad1_nonans.pt')
        torch.save(total_kernel, './nanconv_unittests/kernel_stride3_pad1_nonans.pt')
        torch.save(total_output, f'./nanconv_unittests/conv_{nonan_filename}_stride3_pad1_nonans.pt')


    def prep_mixed_nans(self, mixednan_filename='custom'):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

         [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, 10.]]]]
        """
        torch.manual_seed(0)
   
        n = 1
        total_inputs = []
        total_kernel = []
        total_output = []
        for i in range(n):
            kernel = torch.randn(2, 2, 3, 3)
            inputs = torch.randn(1, 2, 5, 5)

            inputs[:,:, ::2, 0] = float('nan')
            inputs[:,:, ::2, -1] = float('nan')
            # inputs[:,:, -1, :] = float('nan')
            inputs[:,:, 2, :] = float('nan')

            total_kernel.append(kernel)
            total_inputs.append(inputs)

            output = F.conv2d(inputs, kernel, stride=1, padding=1)
            total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, './nanconv_unittests/inputs_mixednans.pt')
        torch.save(total_kernel, './nanconv_unittests/kernel_mixednans.pt')
        torch.save(total_output, f'./nanconv_unittests/conv_{mixednan_filename}_mixednans.pt')


class TestNaNConv2d(unittest.TestCase):
    
    def setUp(self):
        # Default values
        self.nonan_fault_tolerance = getattr(self, 'nonan_fault_tolerance', 0.00005)
        self.mixed_fault_tolerance = getattr(self, 'mixed_fault_tolerance', 0.00005)
        self.save_dir = getattr(self, 'save_dir', '/nanconv_unittests')

    def test_multichannel(self):
        """
        Example Input:
        [[[[ 1.,  2.,  3.,  4.],
          [ 5.,  6., nan,  8.],
          [ 9., nan, nan, nan],
          [nan, 14., 15., 16.]],

         [[17., 18., 19., 20.],
          [nan, 22., 23., nan],
          [nan, 26., 27., 28.],
          [29., nan, 31., nan]],

         [[33., nan, 35., 36.],
          [37., 38., 39., nan],
          [41., nan, 43., nan],
          [45., nan, 47., 48.]]]]
        """
        torch.manual_seed(0)
        
        n = 1
        batch_size = 1
        channels = 2
        nan_inputs = torch.load('./nanconv_unittests/inputs_multichannel.pt')
        nan_kernel = torch.ones(1, 1*channels, 2, 2).float()
        # nan_kernel = torch.load('./nanconv_unittests/kernel_multichannel.pt')
        # nan_output = torch.load('./nanconv_unittests/conv_custom_multichannel.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel, stride=1, padding=0))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load('./nanconv_unittests/inputs_default_mixednans.pt')
        # default_kernel = torch.load('./nanconv_unittests/kernel_default_mixednans.pt')
        default_output = torch.load('./nanconv_unittests/conv_default_multichannel.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-10).all(): fail += 1

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv multi channel Test failed: Torch and NaNConv convolutions do not match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv multi channel Test passed: Torch and NaNConv convolutions match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


    def test_multibatch(self):
        """
        Example Input:
        [[[[ 1.,  2.,  3.,  4.],
          [ 5.,  6., nan,  8.],
          [ 9., nan, nan, nan],
          [nan, 14., 15., 16.]]],


        [[[17., 18., 19., 20.],
          [nan, 22., 23., nan],
          [nan, 26., 27., 28.],
          [29., nan, 31., nan]]]
        """
        torch.manual_seed(0)

        n = 1
        channels = 1
        batch_size = 5
        nan_inputs = torch.load('./nanconv_unittests/inputs_multibatch.pt')
        nan_kernel = torch.ones(1, 1*channels, 2, 2).float()
        # nan_kernel = torch.load('./nanconv_unittests/kernel_multibatch.pt')
        # nan_output = torch.load('./nanconv_unittests/conv_custom_multibatch.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel, stride=1, padding=0))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load('./nanconv_unittests/inputs_default_mixednans.pt')
        # default_kernel = torch.load('./nanconv_unittests/kernel_default_mixednans.pt')
        default_output = torch.load('./nanconv_unittests/conv_default_multibatch.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-10).all(): fail += 1

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv multi batch Test failed: Torch and NaNConv convolutions do not match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv multi batch Test passed: Torch and NaNConv convolutions match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


    def test_multi4d(self):
        """
        Example Input:
        [[[[  1.,   2.,   3.,   4.],
          [  5.,   6.,  nan,   8.],
          [  9.,  nan,  nan,  nan],
          [ nan,  14.,  15.,  16.]],

         [[ 33.,  nan,  35.,  36.],
          [ 37.,  38.,  39.,  nan],
          [ 41.,  nan,  43.,  nan],
          [ 45.,  nan,  47.,  48.]]],


        [[[ nan,  50.,  51.,  52.],
          [ 53.,  54.,  55.,  56.],
          [ 57.,  nan,  59.,  nan],
          [ 61.,  nan,  nan,  64.]],

         [[ 81.,  nan,  nan,  84.],
          [ 85.,  86.,  nan,  88.],
          [ 89.,  90.,  nan,  nan],
          [ 93.,  94.,  95.,  96.]]],

        [[[ 97.,  98.,  nan,  nan],
          [101., 102., 103.,  nan],
          [105., 106., 107.,  nan],
          [109., 110.,  nan, 112.]],

         [[129., 130.,  nan,  nan],
          [133.,  nan, 135.,  nan],
          [137., 138., 139., 140.],
          [141., 142.,  nan, 144.]]]])
        """
        torch.manual_seed(0)

        n = 1
        channels = 3
        batch_size = 3
        nan_inputs = torch.load('./nanconv_unittests/inputs_multi4d.pt')
        nan_kernel = torch.ones(1, 1*channels, 2, 2).float()
        # nan_kernel = torch.load('./nanconv_unittests/kernel_multi4d.pt')
        # nan_output = torch.load('./nanconv_unittests/conv_custom_multi4d.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel, stride=1, padding=0))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load('./nanconv_unittests/inputs_default_mixednans.pt')
        # default_kernel = torch.load('./nanconv_unittests/kernel_default_mixednans.pt')
        default_output = torch.load('./nanconv_unittests/conv_default_multi4d.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-10).all(): fail += 1

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv multi 4D Test failed: Torch and NaNConv convolutions do not match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv multi 4D Test passed: Torch and NaNConv convolutions match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


    def test_all_nans(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]]]
        """
        torch.manual_seed(0)

        kernel = torch.randn(2, 2, 3, 3)
        inputs = torch.randn(1, 2, 5, 5)
        inputs[:] = float('nan')

        output = torch.zeros((1,2,5,5))
        output[:,:, 1:-1, 1:-1] = float('nan')
        mask = torch.isnan(output)
        output[mask] = 1

        nanoutput = F.conv2d(inputs, kernel, stride=1, padding=1)
        # conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
        # nanoutput = conv(inputs)
        nanoutput[mask] = 1
        
        self.assertTrue(torch.equal(output, nanoutput), "NanConv all nans Test failed: Torch and NaNConv values do not match.")
        print("NanConv all nans Test passed: Torch and NaNConv values match.")



    def test_mixed_nans(self):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

          [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

          [[nan, nan],
          [nan, 10.]]]]
        """
        torch.manual_seed(0)
        
        n = 1
        nan_inputs = torch.load('./nanconv_unittests/inputs_mixednans.pt')
        nan_kernel = torch.load('./nanconv_unittests/kernel_mixednans.pt')
        # nan_output = torch.load('./nanconv_unittests/conv_custom_mixednans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=1))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load('./nanconv_unittests/inputs_default_mixednans.pt')
        # default_kernel = torch.load('./nanconv_unittests/kernel_default_mixednans.pt')
        default_output = torch.load('./nanconv_unittests/conv_default_mixednans.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-10).all(): fail += 1
          
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv mixed nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv mixed nan Test passed: Torch and NaNConv inputs match ")

        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv mixed nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv mixed nan Test passed: Torch and NaNConv kernels match ")

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv mixed nan Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=1 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv mixed nan Test passed: Torch and NaNConv convolutions match for padding=1 and stride=1 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


    def test_no_nans(self): 
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

          [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

          [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        torch.manual_seed(0)
        n = 1

        #STRIDE = 1 & PADDING = 1
        nan_inputs = torch.load('./nanconv_unittests/inputs_stride1_pad1_nonans.pt')
        nan_kernel = torch.load('./nanconv_unittests/kernel_stride1_pad1_nonans.pt')
        # nan_output = torch.load('./nanconv_unittests/conv_custom_stride1_pad1_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=1))

        nan_output = torch.stack(nan_output)

        default_output = torch.load('./nanconv_unittests/conv_default_stride1_pad1_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-09).all(): fail += 1
        
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv no nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv inputs match ")
        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv no nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv kernels match ")
        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        #STRIDE = 1 & PADDING = 0
        nan_inputs = torch.load('./nanconv_unittests/inputs_stride1_pad0_nonans.pt')
        nan_kernel = torch.load('./nanconv_unittests/kernel_stride1_pad0_nonans.pt')
        # nan_output = torch.load('./nanconv_unittests/conv_custom_stride1_pad0_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=0))

        nan_output = torch.stack(nan_output)
        default_output = torch.load('./nanconv_unittests/conv_default_stride1_pad0_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-09).all(): fail += 1
        
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv no nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv inputs match ")
        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv no nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv kernels match ")
        # Test conv output
        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=0 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=0 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        #STRIDE = 3 & PADDING = 1
        nan_inputs = torch.load('./nanconv_unittests/inputs_stride3_pad1_nonans.pt')
        nan_kernel = torch.load('./nanconv_unittests/kernel_stride3_pad1_nonans.pt')
        # nan_output = torch.load('./nanconv_unittests/conv_custom_stride3_pad1_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=3, padding=1))

        nan_output = torch.stack(nan_output)

        default_output = torch.load('./nanconv_unittests/conv_default_stride3_pad1_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-09).all(): fail += 1
        
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv no nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv inputs match ")
        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv no nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv kernels match ")
        # Test conv output
        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=3 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=3 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        

  
def run_single_test_with_arg(test_class, test_name, **kwargs):
    suite = unittest.TestSuite()
    test = test_class(test_name)

    # Set custom attributes
    for key, value in kwargs.items():
        setattr(test, key, value)

    suite.addTest(test)
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    # Run all tests
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestNaNPool2dPy)
    # unittest.TextTestRunner().run(suite)

    # suite = unittest.TestLoader().loadTestsFromTestCase(TestNaNConv2d)
    # unittest.TextTestRunner().run(suite)

    # prep = PrepNaNConv2d(save_dir='./nanconv_unittests')
    # prep.prep_mixed_nans(mixednan_filename='default')
    # prep.prep_no_nans(nonan_filename='default')
    # prep.prep_multichannel(multichannel_filename='default')
    # prep.prep_multibatch(multibatch_filename='default')
    # prep.prep_multi4d(multi4d_filename='default')

    # # Run a specific test
    run_single_test_with_arg(TestNaNConv2d, 'test_no_nans', nonan_fault_tolerance=0.1, mixed_fault_tolerance=0.1) #possible to adjust fault tolerance for NaNConv tests