import torch
import torch.nn.functional as F
import numpy as np
import unittest
#from nan_ops import NaNPool2d, NaNConv2d
import os

class TestNaNPool2dPy(unittest.TestCase):

    def unit_test_indices(self, input_tensor, expected, name, test_torch=True, test_expected=True):
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
        input[:] = np.nan
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
        input[:] = np.nan
        input[0] = 1
        input[0, 0, 0, 0] = np.nan
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
        input = torch.tensor([[[[np.nan, np.nan], [33.,  3.]],
                         [[75., 59.], [53., 24.]]],
                         [[[61., 64.], [26., 27.]],
                          [[67., 68.], [11., np.nan]]]])

        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[2]], [[0]]], [[[1]], [[1]]]])
        expected_val = torch.tensor([[[[33]], [[75]]], [[[64]], [[68]]]])
        self.unit_test_indices(input, expected_idx, name='NaNPool mixed nans no multi max', test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, name='NaNPool mixed nans no multi max', test_torch=False, test_expected=True)


class TestNaNConv2dPy(unittest.TestCase):
    
    def setUp(self):
        # Default values
        self.nonan_fault_tolerance = getattr(self, 'nonan_fault_tolerance', 0.05)
        self.mixed_fault_tolerance = getattr(self, 'mixed_fault_tolerance', 0.02)

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
        
        kernel = torch.randn(8, 4, 3, 3)
        inputs = torch.randn(1, 4, 5, 5)
        inputs[:] = np.nan

        output = torch.zeros((1,8,5,5))
        output[:,:, 1:-1, 1:-1] = np.nan
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
        fail=0
        n = 1000

        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)

          output = F.conv2d(inputs, kernel, stride=1, padding=1)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-04).all(): fail += 1

        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        fail=0
        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=0, stride=1, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-04).all(): fail += 1

        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=0 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=0 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        fail=0
        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)

          output = F.conv2d(inputs, kernel, stride=3, padding=1)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=3, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-04).all(): fail += 1

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
   
        fail = 0
        n = 1000
        for _ in range(n):
            kernel = torch.randn(8, 4, 3, 3)
            inputs = torch.randn(1, 4, 5, 5)

            inputs[:,:, ::2, 0] = np.nan
            inputs[:,:, ::2, -1] = np.nan
            # inputs[:,:, -1, :] = np.nan
            inputs[:,:, 2, :] = np.nan

            output = F.conv2d(inputs, kernel, stride=1, padding=1)

            conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
            nanoutput = conv(inputs)

            mask = torch.isnan(output)
            output[mask] = 1

            nanoutput[mask] = 1
            if not torch.isclose(nanoutput, output, rtol=1e-03).all(): fail += 1
    
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv mixed nan Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=3 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv mixed nan Test passed: Torch and NaNConv convolutions match for padding=1 and stride=3 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


class PrepNaNConv2d:
    
    def __init__(self, save_dir='./'):
       self.save_dir = save_dir
    
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
        # os.environ['THRESHOLD'] = '4'

        n = 1000
        
        total_inputs = []
        total_kernel = []
        total_output = []

        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=1)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_stride1_pad1_nonans.pt')
        torch.save(total_kernel, f'{self.save_dir}/kernel_stride1_pad1_nonans.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{nonan_filename}_stride1_pad1_nonans.pt')


        total_inputs = []
        total_kernel = []
        total_output = []
        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)
          
          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_stride1_pad0_nonans.pt')
        torch.save(total_kernel, f'{self.save_dir}/kernel_stride1_pad0_nonans.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{nonan_filename}_stride1_pad0_nonans.pt')

        
        total_inputs = []
        total_kernel = []
        total_output = []
        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)
          
          output = F.conv2d(inputs, kernel, stride=3, padding=1)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_stride3_pad1_nonans.pt')
        torch.save(total_kernel, f'{self.save_dir}/kernel_stride3_pad1_nonans.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{nonan_filename}_stride3_pad1_nonans.pt')


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
        # os.environ['THRESHOLD'] = '4'
   
        n = 1000
        total_inputs = []
        total_kernel = []
        total_output = []
        for i in range(n):
            kernel = torch.randn(8, 4, 3, 3)
            inputs = torch.randn(1, 4, 5, 5)

            inputs[:,:, ::2, 0] = np.nan
            inputs[:,:, ::2, -1] = np.nan
            # inputs[:,:, -1, :] = np.nan
            inputs[:,:, 2, :] = np.nan

            total_kernel.append(kernel)
            total_inputs.append(inputs)

            output = F.conv2d(inputs, kernel, stride=1, padding=1)
            total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_mixednans.pt')
        torch.save(total_kernel, f'{self.save_dir}/kernel_mixednans.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{mixednan_filename}_mixednans.pt')


class TestNaNConv2d(unittest.TestCase):
    
    def setUp(self):
        # Default values
        self.nonan_fault_tolerance = getattr(self, 'nonan_fault_tolerance', 0.05)
        self.mixed_fault_tolerance = getattr(self, 'mixed_fault_tolerance', 0.02)
        self.save_dir = getattr(self, 'save_dir', '/nanconv_unittests')
    
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
        os.environ['THRESHOLD'] = '9'

        kernel = torch.randn(8, 4, 3, 3)
        inputs = torch.randn(1, 4, 5, 5)
        inputs[:] = np.nan

        output = torch.zeros((1,8,5,5))
        output[:,:, 1:-1, 1:-1] = np.nan
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
        os.environ['THRESHOLD'] = '9'
        
        n = 1000
        nan_inputs = torch.load(f'{self.save_dir}/inputs_mixednans.pt')
        nan_kernel = torch.load(f'{self.save_dir}/kernel_mixednans.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_mixednans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=1))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load(f'{self.save_dir}/inputs_default_mixednans.pt')
        # default_kernel = torch.load(f'{self.save_dir}/kernel_default_mixednans.pt')
        default_output = torch.load(f'{self.save_dir}/conv_default_mixednans.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-03).all(): fail += 1
          
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv mixed nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv mixed nan Test passed: Torch and NaNConv inputs match ")

        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv mixed nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv mixed nan Test passed: Torch and NaNConv kernels match ")

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv mixed nan Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=3 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
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
        n = 1000
        os.environ['THRESHOLD'] = '9'

        #STRIDE = 1 & PADDING = 1
        nan_inputs = torch.load(f'{self.save_dir}/inputs_stride1_pad1_nonans.pt')
        nan_kernel = torch.load(f'{self.save_dir}/kernel_stride1_pad1_nonans.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_stride1_pad1_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=1))

        nan_output = torch.stack(nan_output)

        default_output = torch.load(f'{self.save_dir}/conv_default_stride1_pad1_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-04).all(): fail += 1
        
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
        nan_inputs = torch.load(f'{self.save_dir}/inputs_stride1_pad0_nonans.pt')
        nan_kernel = torch.load(f'{self.save_dir}/kernel_stride1_pad0_nonans.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_stride1_pad0_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=0))

        nan_output = torch.stack(nan_output)
        default_output = torch.load(f'{self.save_dir}/conv_default_stride1_pad0_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-04).all(): fail += 1
        
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
        nan_inputs = torch.load(f'{self.save_dir}/inputs_stride3_pad1_nonans.pt')
        nan_kernel = torch.load(f'{self.save_dir}/kernel_stride3_pad1_nonans.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_stride3_pad1_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=0))

        nan_output = torch.stack(nan_output)

        default_output = torch.load(f'{self.save_dir}/conv_default_stride3_pad1_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-04).all(): fail += 1
        
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
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestNaNPool2d)
    # unittest.TextTestRunner().run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestNaNConv2d)
    unittest.TextTestRunner().run(suite)

    prep = PrepNaNConv2d(save_dir='../workspace')
    prep.prep_mixed_nans(mixednan_filename='default')
    prep.prep_no_nans(nonan_filename='default')


    # # Run a specific test
    # run_single_test_with_arg(TestNaNConv2d, 'test_mixed_nans', nonan_fault_tolerance=0.1, mixed_fault_tolerance=0.1) #possible to adjust fault tolerance for NaNConv tests