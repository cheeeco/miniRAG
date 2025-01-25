# **NumPy Reference**

*Release 2.2.0*

**Written by the NumPy community**

**January 19, 2025**

# **CONTENTS**

| 1 | Python API | 3 |
| --- | --- | --- |
| 2 | C API | 1875 |
| 3 | Other topics | 2003 |
| 4 | Acknowledgements | 2057 |
| Bibliography |  | 2059 |
| Python Module Index |  | 2073 |

# **Release** 2.2

**Date** January 19, 2025

This reference manual details functions, modules, and objects included in NumPy, describing what they are and what they do. For learning how to use NumPy, see the complete documentation.

### **CHAPTER**

# **ONE**

# **PYTHON API**

# **1.1 NumPy's module structure**

NumPy has a large number of submodules. Most regular usage of NumPy requires only the main namespace and a smaller set of submodules. The rest either either special-purpose or niche namespaces.

# **1.1.1 Main namespaces**

Regular/recommended user-facing namespaces for general use:

- *numpy*
- *numpy.exceptions*
- *numpy.fft*
- *numpy.linalg*
- *numpy.polynomial*
- *numpy.random*
- *numpy.strings*
- *numpy.testing*
- *numpy.typing*

# **1.1.2 Special-purpose namespaces**

- *numpy.ctypeslib* interacting with NumPy objects with ctypes
- *numpy.dtypes* dtype classes (typically not used directly by end users)
- *numpy.emath* mathematical functions with automatic domain
- *numpy.lib* utilities & functionality which do not fit the main namespace
- *numpy.rec* record arrays (largely superseded by dataframe libraries)
- *numpy.version* small module with more detailed version info

# **1.1.3 Legacy namespaces**

Prefer not to use these namespaces for new code. There are better alternatives and/or this code is deprecated or isn't reliable.

- *numpy.char* legacy string functionality, only for fixed-width strings
- *numpy.distutils* (deprecated) build system support
- *numpy.f2py* Fortran binding generation (usually used from the command line only)
- *numpy.ma* masked arrays (not very reliable, needs an overhaul)
- *numpy.matlib* (pending deprecation) functions supporting matrix instances

### **Exceptions and Warnings (numpy.exceptions)**

General exceptions used by NumPy. Note that some exceptions may be module specific, such as linear algebra errors.

New in version NumPy: 1.25

The exceptions module is new in NumPy 1.25. Older exceptions remain available through the main NumPy namespace for compatibility.

### **Warnings**

| ComplexWarning | The warning raised when casting a complex dtype to a real |
| --- | --- |
|  | dtype. |
| VisibleDeprecationWarning | Visible deprecation warning. |
| RankWarning | Matrix rank warning. |

### **exception** exceptions.**ComplexWarning**

The warning raised when casting a complex dtype to a real dtype.

As implemented, casting a complex number to a real discards its imaginary part, but this behavior may not be what the user actually wants.

### **exception** exceptions.**VisibleDeprecationWarning**

Visible deprecation warning.

By default, python will not show deprecation warnings, so this class can be used when a very visible warning is helpful, for example because the usage is most likely a user bug.

### **exception** exceptions.**RankWarning**

Matrix rank warning.

Issued by polynomial functions when the design matrix is rank deficient.

#### **Exceptions**

| AxisError(axis[, ndim, msg_prefix]) | Axis supplied was invalid. |
| --- | --- |
| DTypePromotionError | Multiple DTypes could not be converted to a common |
|  | one. |
| TooHardError | max_work was exceeded. |

**exception** exceptions.**AxisError**(*axis*, *ndim=None*, *msg_prefix=None*)

Axis supplied was invalid.

This is raised whenever an axis parameter is specified that is larger than the number of array dimensions. For compatibility with code written against older numpy versions, which raised a mixture of ValueError and IndexError for this situation, this exception subclasses both to ensure that except ValueError and except IndexError statements continue to catch AxisError.

#### **Parameters**

#### **axis**

[int or str] The out of bounds axis or a custom exception message. If an axis is provided, then *ndim* should be specified as well.

#### **ndim**

[int, optional] The number of array dimensions.

#### **msg_prefix**

[str, optional] A prefix for the exception message.

### **Examples**

```
>>> import numpy as np
>>> array_1d = np.arange(10)
>>> np.cumsum(array_1d, axis=1)
Traceback (most recent call last):
  ...
numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1
```
Negative axes are preserved:

```
>>> np.cumsum(array_1d, axis=-2)
Traceback (most recent call last):
  ...
numpy.exceptions.AxisError: axis -2 is out of bounds for array of dimension 1
```
The class constructor generally takes the axis and arrays' dimensionality as arguments:

```
>>> print(np.exceptions.AxisError(2, 1, msg_prefix='error'))
error: axis 2 is out of bounds for array of dimension 1
```
Alternatively, a custom exception message can be passed:

```
>>> print(np.exceptions.AxisError('Custom error message'))
Custom error message
```
**Attributes**

#### **axis**

[int, optional] The out of bounds axis or None if a custom exception message was provided. This should be the axis as passed by the user, before any normalization to resolve negative indices.

New in version 1.22.

### **ndim**

[int, optional] The number of array dimensions or None if a custom exception message was provided.

New in version 1.22.

### **exception** exceptions.**DTypePromotionError**

Multiple DTypes could not be converted to a common one.

This exception derives from TypeError and is raised whenever dtypes cannot be converted to a single common one. This can be because they are of a different category/class or incompatible instances of the same one (see Examples).

#### **Notes**

Many functions will use promotion to find the correct result and implementation. For these functions the error will typically be chained with a more specific error indicating that no implementation was found for the input dtypes.

Typically promotion should be considered "invalid" between the dtypes of two arrays when *arr1 == arr2* can safely return all False because the dtypes are fundamentally different.

### **Examples**

Datetimes and complex numbers are incompatible classes and cannot be promoted:

```
>>> import numpy as np
>>> np.result_type(np.dtype("M8[s]"), np.complex128)
Traceback (most recent call last):
 ...
DTypePromotionError: The DType <class 'numpy.dtype[datetime64]'> could not
be promoted by <class 'numpy.dtype[complex128]'>. This means that no common
DType exists for the given inputs. For example they cannot be stored in a
single array unless the dtype is `object`. The full list of DTypes is:
(<class 'numpy.dtype[datetime64]'>, <class 'numpy.dtype[complex128]'>)
```
For example for structured dtypes, the structure can mismatch and the same DTypePromotionError is given when two structured dtypes with a mismatch in their number of fields is given:

```
>>> dtype1 = np.dtype([("field1", np.float64), ("field2", np.int64)])
>>> dtype2 = np.dtype([("field1", np.float64)])
>>> np.promote_types(dtype1, dtype2)
Traceback (most recent call last):
 ...
DTypePromotionError: field names `('field1', 'field2')` and `('field1',)`
mismatch.
```
**exception** exceptions.**TooHardError**

max_work was exceeded.

This is raised whenever the maximum number of candidate solutions to consider specified by the max_work parameter is exceeded. Assigning a finite number to max_work may have caused the operation to fail.

#### **Discrete Fourier Transform (numpy.fft)**

The SciPy module scipy.fft is a more comprehensive superset of numpy.fft, which includes only a basic set of routines.

#### **Standard FFTs**

| fft(a[, n, axis, norm, out]) | Compute the one-dimensional discrete Fourier Trans |
| --- | --- |
|  | form. |
| ifft(a[, n, axis, norm, out]) | Compute the one-dimensional inverse discrete Fourier |
|  | Transform. |
| fft2(a[, s, axes, norm, out]) | Compute the 2-dimensional discrete Fourier Transform. |
| ifft2(a[, s, axes, norm, out]) | Compute the 2-dimensional inverse discrete Fourier |
|  | Transform. |
| fftn(a[, s, axes, norm, out]) | Compute the N-dimensional discrete Fourier Transform. |
| ifftn(a[, s, axes, norm, out]) | Compute the N-dimensional inverse discrete Fourier |
|  | Transform. |

#### fft.**fft**(*a*, *n=None*, *axis=-1*, *norm=None*, *out=None*)

Compute the one-dimensional discrete Fourier Transform.

This function computes the one-dimensional *n*-point discrete Fourier Transform (DFT) with the efficient Fast Fourier Transform (FFT) algorithm [CT].

### **Parameters**

#### **a**

[array_like] Input array, can be complex.

### **n**

[int, optional] Length of the transformed axis of the output. If *n* is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros. If *n* is not given, the length of the input along the axis specified by *axis* is used.

#### **axis**

[int, optional] Axis over which to compute the FFT. If not given, the last axis is used.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype.

New in version 2.0.0.

#### **Returns**

#### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axis indicated by *axis*, or the last one if *axis* is not specified.

#### **Raises**

**IndexError**

If *axis* is not a valid axis of *a*.

### **See also:**

#### *numpy.fft*

for definition of the DFT and conventions used.

### *ifft*

The inverse of *fft*.

### *fft2*

The two-dimensional FFT.

### *fftn*

The *n*-dimensional FFT.

### *rfftn*

The *n*-dimensional FFT of real input.

### *fftfreq*

Frequency bins for given FFT parameters.

### **Notes**

FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform (DFT) can be calculated efficiently, by using symmetries in the calculated terms. The symmetry is highest when *n* is a power of 2, and the transform is therefore most efficient for these sizes.

The DFT is defined, with the conventions used in this implementation, in the documentation for the *numpy.fft* module.

### **References**

[CT]

### **Examples**

```
>>> import numpy as np
>>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
array([-2.33486982e-16+1.14423775e-17j, 8.00000000e+00-1.25557246e-15j,
        2.33486982e-16+2.33486982e-16j, 0.00000000e+00+1.22464680e-16j,
       -1.14423775e-17+2.33486982e-16j, 0.00000000e+00+5.20784380e-16j,
        1.14423775e-17+1.14423775e-17j, 0.00000000e+00+1.22464680e-16j])
```
In this example, real input has an FFT which is Hermitian, i.e., symmetric in the real part and anti-symmetric in the imaginary part, as described in the *numpy.fft* documentation:

```
>>> import matplotlib.pyplot as plt
>>> t = np.arange(256)
>>> sp = np.fft.fft(np.sin(t))
>>> freq = np.fft.fftfreq(t.shape[-1])
>>> plt.plot(freq, sp.real, freq, sp.imag)
[<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x.
```
(continues on next page)

(continued from previous page)

*,→*..>] **>>>** plt.show()

![](_page_12_Figure_3.jpeg)

fft.**ifft**(*a*, *n=None*, *axis=-1*, *norm=None*, *out=None*)

Compute the one-dimensional inverse discrete Fourier Transform.

This function computes the inverse of the one-dimensional *n*-point discrete Fourier transform computed by *fft*. In other words, ifft(fft(a)) == a to within numerical accuracy. For a general description of the algorithm and definitions, see *numpy.fft*.

The input should be ordered in the same way as is returned by *fft*, i.e.,

- a[0] should contain the zero frequency term,
- a[1:n//2] should contain the positive-frequency terms,
- a[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.

For an even number of input points, A[n//2] represents the sum of the values at the positive and negative Nyquist frequencies, as the two are aliased together. See *numpy.fft* for details.

#### **Parameters**

**a**

[array_like] Input array, can be complex.

**n**

[int, optional] Length of the transformed axis of the output. If *n* is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros. If *n* is not given, the length of the input along the axis specified by *axis* is used. See notes about padding issues.

#### **axis**

[int, optional] Axis over which to compute the inverse DFT. If not given, the last axis is used.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype.

New in version 2.0.0.

### **Returns**

### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axis indicated by *axis*, or the last one if *axis* is not specified.

### **Raises**

#### **IndexError**

If *axis* is not a valid axis of *a*.

#### **See also:**

#### *numpy.fft*

An introduction, with definitions and general explanations.

#### *fft*

The one-dimensional (forward) FFT, of which *ifft* is the inverse

*ifft2*

The two-dimensional inverse FFT.

#### *ifftn*

The n-dimensional inverse FFT.

#### **Notes**

If the input parameter *n* is larger than the size of the input, the input is padded by appending zeros at the end. Even though this is the common approach, it might lead to surprising results. If a different padding is desired, it must be performed before calling *ifft*.

#### **Examples**

```
>>> import numpy as np
>>> np.fft.ifft([0, 4, 0, 0])
array([ 1.+0.j, 0.+1.j, -1.+0.j, 0.-1.j]) # may vary
```
Create and plot a band-limited signal with random phases:

```
>>> import matplotlib.pyplot as plt
>>> t = np.arange(400)
>>> n = np.zeros((400,), dtype=complex)
>>> n[40:60] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20,)))
>>> s = np.fft.ifft(n)
>>> plt.plot(t, s.real, label='real')
[<matplotlib.lines.Line2D object at ...>]
>>> plt.plot(t, s.imag, '--', label='imaginary')
[<matplotlib.lines.Line2D object at ...>]
>>> plt.legend()
```
(continues on next page)

(continued from previous page)

![](_page_14_Figure_2.jpeg)

![](_page_14_Figure_3.jpeg)

#### fft.**fft2**(*a*, *s=None*, *axes=(-2, -1)*, *norm=None*, *out=None*)

Compute the 2-dimensional discrete Fourier Transform.

This function computes the *n*-dimensional discrete Fourier Transform over any axes in an *M*-dimensional array by means of the Fast Fourier Transform (FFT). By default, the transform is computed over the last two axes of the input array, i.e., a 2-dimensional FFT.

#### **Parameters**

**a**

[array_like] Input array, can be complex

**s**

[sequence of ints, optional] Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). This corresponds to n for fft(x, n). Along each axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros.

Changed in version 2.0: If it is -1, the whole input is used (no padding/trimming).

If *s* is not given, the shape of the input along the axes specified by *axes* is used.

Deprecated since version 2.0: If *s* is not None, *axes* must not be None either.

Deprecated since version 2.0: *s* must contain only int s, not None values. None values currently mean that the default value for n is used in the corresponding 1-D transform, but this behaviour is deprecated.

#### **axes**

[sequence of ints, optional] Axes over which to compute the FFT. If not given, the last two axes are used. A repeated index in *axes* means the transform over that axis is performed multiple times. A one-element sequence means that a one-dimensional FFT is performed. Default: (-2, -1).

Deprecated since version 2.0: If *s* is specified, the corresponding *axes* to be transformed must not be None.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype for all axes (and hence only the last axis can have s not equal to the shape at that axis).

New in version 2.0.0.

#### **Returns**

#### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axes indicated by *axes*, or the last two axes if *axes* is not given.

### **Raises**

### **ValueError**

If *s* and *axes* have different length, or *axes* not given and len(s) != 2.

### **IndexError**

If an element of *axes* is larger than than the number of axes of *a*.

### **See also:**

#### *numpy.fft*

Overall view of discrete Fourier transforms, with definitions and conventions used.

#### *ifft2*

The inverse two-dimensional FFT.

#### *fft*

The one-dimensional FFT.

#### *fftn*

The *n*-dimensional FFT.

#### *fftshift*

Shifts zero-frequency terms to the center of the array. For two-dimensional input, swaps first and third quadrants, and second and fourth quadrants.

#### **Notes**

*fft2* is just *fftn* with a different default for *axes*.

The output, analogously to *fft*, contains the term for zero frequency in the low-order corner of the transformed axes, the positive frequency terms in the first half of these axes, the term for the Nyquist frequency in the middle of the axes and the negative frequency terms in the second half of the axes, in order of decreasingly negative frequency.

See *fftn* for details and a plotting example, and *numpy.fft* for definitions and conventions used.

### **Examples**

| >>> import numpy as np |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| >>> a = np.mgrid[:5, :5][0] |  |  |  |  |  |  |  |  |
| >>> np.fft.fft2(a) |  |  |  |  |  |  |  |  |
| array([[ 50. +0.j |  | , | 0. | +0.j | , | 0. | +0.j | , # may vary |
| 0. | +0.j | , | 0. | +0.j | ], |  |  |  |
| [-12.5+17.20477401j, |  |  | 0. | +0.j | , | 0. | +0.j | , |
| 0. | +0.j | , | 0. | +0.j | ], |  |  |  |
| [-12.5 +4.0614962j , |  |  | 0. | +0.j | , | 0. | +0.j | , |
| 0. | +0.j | , | 0. | +0.j | ], |  |  |  |
| [-12.5 -4.0614962j , |  |  | 0. | +0.j | , | 0. | +0.j | , |
| 0. | +0.j | , | 0. | +0.j | ], |  |  |  |
| [-12.5-17.20477401j, |  |  | 0. | +0.j | , | 0. | +0.j | , |
| 0. | +0.j | , | 0. | +0.j | ]]) |  |  |  |

### fft.**ifft2**(*a*, *s=None*, *axes=(-2, -1)*, *norm=None*, *out=None*)

Compute the 2-dimensional inverse discrete Fourier Transform.

This function computes the inverse of the 2-dimensional discrete Fourier Transform over any number of axes in an M-dimensional array by means of the Fast Fourier Transform (FFT). In other words, ifft2(fft2(a)) == a to within numerical accuracy. By default, the inverse transform is computed over the last two axes of the input array.

The input, analogously to *ifft*, should be ordered in the same way as is returned by *fft2*, i.e. it should have the term for zero frequency in the low-order corner of the two axes, the positive frequency terms in the first half of these axes, the term for the Nyquist frequency in the middle of the axes and the negative frequency terms in the second half of both axes, in order of decreasingly negative frequency.

### **Parameters**

**a**

[array_like] Input array, can be complex.

**s**

[sequence of ints, optional] Shape (length of each axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). This corresponds to *n* for ifft(x, n). Along each axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros.

Changed in version 2.0: If it is -1, the whole input is used (no padding/trimming).

If *s* is not given, the shape of the input along the axes specified by *axes* is used. See notes for issue on *ifft* zero padding.

Deprecated since version 2.0: If *s* is not None, *axes* must not be None either.

Deprecated since version 2.0: *s* must contain only int s, not None values. None values currently mean that the default value for n is used in the corresponding 1-D transform, but this behaviour is deprecated.

### **axes**

[sequence of ints, optional] Axes over which to compute the FFT. If not given, the last two axes are used. A repeated index in *axes* means the transform over that axis is performed multiple times. A one-element sequence means that a one-dimensional FFT is performed. Default: (-2, -1).

Deprecated since version 2.0: If *s* is specified, the corresponding *axes* to be transformed must not be None.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype for all axes (and hence is incompatible with passing in all but the trivial s).

New in version 2.0.0.

### **Returns**

#### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axes indicated by *axes*, or the last two axes if *axes* is not given.

#### **Raises**

#### **ValueError**

If *s* and *axes* have different length, or *axes* not given and len(s) != 2.

#### **IndexError**

If an element of *axes* is larger than than the number of axes of *a*.

### **See also:**

#### *numpy.fft*

Overall view of discrete Fourier transforms, with definitions and conventions used.

#### *fft2*

The forward 2-dimensional FFT, of which *ifft2* is the inverse.

#### *ifftn*

The inverse of the *n*-dimensional FFT.

#### *fft*

The one-dimensional FFT.

#### *ifft*

The one-dimensional inverse FFT.

### **Notes**

*ifft2* is just *ifftn* with a different default for *axes*.

See *ifftn* for details and a plotting example, and *numpy.fft* for definition and conventions used.

Zero-padding, analogously with *ifft*, is performed by appending zeros to the input along the specified dimension. Although this is the common approach, it might lead to surprising results. If another form of zero padding is desired, it must be performed before *ifft2* is called.

#### **Examples**

```
>>> import numpy as np
>>> a = 4 * np.eye(4)
>>> np.fft.ifft2(a)
array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], # may vary
      [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
      [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
      [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
```
### fft.**fftn**(*a*, *s=None*, *axes=None*, *norm=None*, *out=None*)

Compute the N-dimensional discrete Fourier Transform.

This function computes the *N*-dimensional discrete Fourier Transform over any number of axes in an *M*dimensional array by means of the Fast Fourier Transform (FFT).

#### **Parameters**

#### **a**

[array_like] Input array, can be complex.

### **s**

[sequence of ints, optional] Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). This corresponds to n for fft(x, n). Along any axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros.

Changed in version 2.0: If it is -1, the whole input is used (no padding/trimming).

If *s* is not given, the shape of the input along the axes specified by *axes* is used.

Deprecated since version 2.0: If *s* is not None, *axes* must not be None either.

Deprecated since version 2.0: *s* must contain only int s, not None values. None values currently mean that the default value for n is used in the corresponding 1-D transform, but this behaviour is deprecated.

#### **axes**

[sequence of ints, optional] Axes over which to compute the FFT. If not given, the last len(s) axes are used, or all axes if *s* is also not specified. Repeated indices in *axes* means that the transform over that axis is performed multiple times.

Deprecated since version 2.0: If *s* is specified, the corresponding *axes* to be transformed must be explicitly specified too.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype for all axes (and hence is incompatible with passing in all but the trivial s).

New in version 2.0.0.

#### **Returns**

### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axes indicated by *axes*, or by a combination of *s* and *a*, as explained in the parameters section above.

#### **Raises**

#### **ValueError**

If *s* and *axes* have different length.

### **IndexError**

If an element of *axes* is larger than than the number of axes of *a*.

### **See also:**

### *numpy.fft*

Overall view of discrete Fourier transforms, with definitions and conventions used.

#### *ifftn*

The inverse of *fftn*, the inverse *n*-dimensional FFT.

#### *fft*

The one-dimensional FFT, with definitions and conventions used.

#### *rfftn*

The *n*-dimensional FFT of real input.

#### *fft2*

The two-dimensional FFT.

### *fftshift*

Shifts zero-frequency terms to centre of array

#### **Notes**

The output, analogously to *fft*, contains the term for zero frequency in the low-order corner of all axes, the positive frequency terms in the first half of all axes, the term for the Nyquist frequency in the middle of all axes and the negative frequency terms in the second half of all axes, in order of decreasingly negative frequency.

See *numpy.fft* for details, definitions and conventions used.

#### **Examples**

```
>>> import numpy as np
>>> a = np.mgrid[:3, :3, :3][0]
>>> np.fft.fftn(a, axes=(1, 2))
array([[[ 0.+0.j, 0.+0.j, 0.+0.j], # may vary
       [ 0.+0.j, 0.+0.j, 0.+0.j],
       [ 0.+0.j, 0.+0.j, 0.+0.j]],
      [[ 9.+0.j, 0.+0.j, 0.+0.j],
       [ 0.+0.j, 0.+0.j, 0.+0.j],
       [ 0.+0.j, 0.+0.j, 0.+0.j]],
      [[18.+0.j, 0.+0.j, 0.+0.j],
       [ 0.+0.j, 0.+0.j, 0.+0.j],
       [ 0.+0.j, 0.+0.j, 0.+0.j]]])
>>> np.fft.fftn(a, (2, 2), axes=(0, 1))
array([[[ 2.+0.j, 2.+0.j, 2.+0.j], # may vary
       [ 0.+0.j, 0.+0.j, 0.+0.j]],
```
(continues on next page)

(continued from previous page)

```
[[-2.+0.j, -2.+0.j, -2.+0.j],
       [ 0.+0.j, 0.+0.j, 0.+0.j]]])
>>> import matplotlib.pyplot as plt
>>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
... 2 * np.pi * np.arange(200) / 34)
>>> S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)
>>> FS = np.fft.fftn(S)
>>> plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))
<matplotlib.image.AxesImage object at 0x...>
>>> plt.show()
```
![](_page_20_Figure_3.jpeg)

### fft.**ifftn**(*a*, *s=None*, *axes=None*, *norm=None*, *out=None*)

Compute the N-dimensional inverse discrete Fourier Transform.

This function computes the inverse of the N-dimensional discrete Fourier Transform over any number of axes in an M-dimensional array by means of the Fast Fourier Transform (FFT). In other words, ifftn(fftn(a)) == a to within numerical accuracy. For a description of the definitions and conventions used, see *numpy.fft*.

The input, analogously to *ifft*, should be ordered in the same way as is returned by *fftn*, i.e. it should have the term for zero frequency in all axes in the low-order corner, the positive frequency terms in the first half of all axes, the term for the Nyquist frequency in the middle of all axes and the negative frequency terms in the second half of all axes, in order of decreasingly negative frequency.

### **Parameters**

**a**

[array_like] Input array, can be complex.

**s**

[sequence of ints, optional] Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). This corresponds to n for ifft(x, n). Along any axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros.

Changed in version 2.0: If it is -1, the whole input is used (no padding/trimming).

If *s* is not given, the shape of the input along the axes specified by *axes* is used. See notes for issue on *ifft* zero padding.

Deprecated since version 2.0: If *s* is not None, *axes* must not be None either.

Deprecated since version 2.0: *s* must contain only int s, not None values. None values currently mean that the default value for n is used in the corresponding 1-D transform, but this behaviour is deprecated.

#### **axes**

[sequence of ints, optional] Axes over which to compute the IFFT. If not given, the last len(s) axes are used, or all axes if *s* is also not specified. Repeated indices in *axes* means that the inverse transform over that axis is performed multiple times.

Deprecated since version 2.0: If *s* is specified, the corresponding *axes* to be transformed must be explicitly specified too.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype for all axes (and hence is incompatible with passing in all but the trivial s).

New in version 2.0.0.

### **Returns**

### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axes indicated by *axes*, or by a combination of *s* or *a*, as explained in the parameters section above.

#### **Raises**

#### **ValueError**

If *s* and *axes* have different length.

### **IndexError**

If an element of *axes* is larger than than the number of axes of *a*.

#### **See also:**

#### *numpy.fft*

Overall view of discrete Fourier transforms, with definitions and conventions used.

### *fftn*

The forward *n*-dimensional FFT, of which *ifftn* is the inverse.

### *ifft*

The one-dimensional inverse FFT.

#### *ifft2*

The two-dimensional inverse FFT.

### *ifftshift*

Undoes *fftshift*, shifts zero-frequency terms to beginning of array.

### **Notes**

See *numpy.fft* for definitions and conventions used.

Zero-padding, analogously with *ifft*, is performed by appending zeros to the input along the specified dimension. Although this is the common approach, it might lead to surprising results. If another form of zero padding is desired, it must be performed before *ifftn* is called.

### **Examples**

```
>>> import numpy as np
>>> a = np.eye(4)
>>> np.fft.ifftn(np.fft.fftn(a, axes=(0,)), axes=(1,))
array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], # may vary
      [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
      [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
      [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
```
Create and plot an image with band-limited frequency content:

```
>>> import matplotlib.pyplot as plt
>>> n = np.zeros((200,200), dtype=complex)
>>> n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))
>>> im = np.fft.ifftn(n).real
>>> plt.imshow(im)
<matplotlib.image.AxesImage object at 0x...>
>>> plt.show()
```
![](_page_22_Figure_8.jpeg)

### **Real FFTs**

| rfft(a[, n, axis, norm, out]) | Compute the one-dimensional discrete Fourier Transform |
| --- | --- |
|  | for real input. |
| irfft(a[, n, axis, norm, out]) | Computes the inverse of rfft. |
| rfft2(a[, s, axes, norm, out]) | Compute the 2-dimensional FFT of a real array. |
| irfft2(a[, s, axes, norm, out]) | Computes the inverse of rfft2. |
| rfftn(a[, s, axes, norm, out]) | Compute the N-dimensional discrete Fourier Transform |
|  | for real input. |
| irfftn(a[, s, axes, norm, out]) | Computes the inverse of rfftn. |

#### fft.**rfft**(*a*, *n=None*, *axis=-1*, *norm=None*, *out=None*)

Compute the one-dimensional discrete Fourier Transform for real input.

This function computes the one-dimensional *n*-point discrete Fourier Transform (DFT) of a real-valued array by means of an efficient algorithm called the Fast Fourier Transform (FFT).

#### **Parameters**

#### **a**

[array_like] Input array

### **n**

[int, optional] Number of points along transformation axis in the input to use. If *n* is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros. If *n* is not given, the length of the input along the axis specified by *axis* is used.

#### **axis**

[int, optional] Axis over which to compute the FFT. If not given, the last axis is used.

### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype.

New in version 2.0.0.

### **Returns**

### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axis indicated by *axis*, or the last one if *axis* is not specified. If *n* is even, the length of the transformed axis is (n/2)+1. If *n* is odd, the length is (n+1)/2.

### **Raises**

### **IndexError**

If *axis* is not a valid axis of *a*.

### **See also:**

#### *numpy.fft*

For definition of the DFT and conventions used.

### *irfft*

The inverse of *rfft*.

#### *fft*

The one-dimensional FFT of general (complex) input.

### *fftn*

The *n*-dimensional FFT.

### *rfftn*

The *n*-dimensional FFT of real input.

#### **Notes**

When the DFT is computed for purely real input, the output is Hermitian-symmetric, i.e. the negative frequency terms are just the complex conjugates of the corresponding positive-frequency terms, and the negative-frequency terms are therefore redundant. This function does not compute the negative frequency terms, and the length of the transformed axis of the output is therefore n//2 + 1.

When A = rfft(a) and fs is the sampling frequency, A[0] contains the zero-frequency term 0*fs, which is real due to Hermitian symmetry.

If *n* is even, A[-1] contains the term representing both positive and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely real. If *n* is odd, there is no term at fs/2; A[-1] contains the largest positive frequency (fs/2*(n-1)/n), and is complex in the general case.

If the input *a* contains an imaginary part, it is silently discarded.

### **Examples**

```
>>> import numpy as np
>>> np.fft.fft([0, 1, 0, 0])
array([ 1.+0.j, 0.-1.j, -1.+0.j, 0.+1.j]) # may vary
>>> np.fft.rfft([0, 1, 0, 0])
array([ 1.+0.j, 0.-1.j, -1.+0.j]) # may vary
```
Notice how the final element of the *fft* output is the complex conjugate of the second element, for real input. For *rfft*, this symmetry is exploited to compute only the non-negative frequency terms.

#### fft.**irfft**(*a*, *n=None*, *axis=-1*, *norm=None*, *out=None*)

Computes the inverse of *rfft*.

This function computes the inverse of the one-dimensional *n*-point discrete Fourier Transform of real input computed by *rfft*. In other words, irfft(rfft(a), len(a)) == a to within numerical accuracy. (See Notes below for why len(a) is necessary here.)

The input is expected to be in the form returned by *rfft*, i.e. the real zero-frequency term followed by the complex positive frequency terms in order of increasing frequency. Since the discrete Fourier Transform of real input is Hermitian-symmetric, the negative frequency terms are taken to be the complex conjugates of the corresponding positive frequency terms.

#### **Parameters**

**a**

[array_like] The input array.

**n**

[int, optional] Length of the transformed axis of the output. For *n* output points, n//2+1

input points are necessary. If the input is longer than this, it is cropped. If it is shorter than this, it is padded with zeros. If *n* is not given, it is taken to be 2*(m-1) where m is the length of the input along the axis specified by *axis*.

#### **axis**

[int, optional] Axis over which to compute the inverse FFT. If not given, the last axis is used.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype.

New in version 2.0.0.

#### **Returns**

#### **out**

[ndarray] The truncated or zero-padded input, transformed along the axis indicated by *axis*, or the last one if *axis* is not specified. The length of the transformed axis is *n*, or, if *n* is not given, 2*(m-1) where m is the length of the transformed axis of the input. To get an odd number of output points, *n* must be specified.

### **Raises**

#### **IndexError**

If *axis* is not a valid axis of *a*.

#### **See also:**

#### *numpy.fft*

For definition of the DFT and conventions used.

#### *rfft*

The one-dimensional FFT of real input, of which *irfft* is inverse.

#### *fft*

The one-dimensional FFT.

#### *irfft2*

The inverse of the two-dimensional FFT of real input.

#### *irfftn*

The inverse of the *n*-dimensional FFT of real input.

#### **Notes**

Returns the real valued *n*-point inverse discrete Fourier transform of *a*, where *a* contains the non-negative frequency terms of a Hermitian-symmetric sequence. *n* is the length of the result, not the input.

If you specify an *n* such that *a* must be zero-padded or truncated, the extra/removed values will be added/removed at high frequencies. One can thus resample a series to *m* points via Fourier interpolation by: a_resamp = irfft(rfft(a), m).

The correct interpretation of the hermitian input depends on the length of the original data, as given by *n*. This is because each input shape could correspond to either an odd or even length signal. By default, *irfft* assumes an

even output length which puts the last entry at the Nyquist frequency; aliasing with its symmetric counterpart. By Hermitian symmetry, the value is thus treated as purely real. To avoid losing information, the correct length of the real input **must** be given.

### **Examples**

```
>>> import numpy as np
>>> np.fft.ifft([1, -1j, -1, 1j])
array([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]) # may vary
>>> np.fft.irfft([1, -1j, -1])
array([0., 1., 0., 0.])
```
Notice how the last term in the input to the ordinary *ifft* is the complex conjugate of the second term, and the output has zero imaginary part everywhere. When calling *irfft*, the negative frequencies are not specified, and the output array is purely real.

#### fft.**rfft2**(*a*, *s=None*, *axes=(-2, -1)*, *norm=None*, *out=None*)

Compute the 2-dimensional FFT of a real array.

# **Parameters a**

[array] Input array, taken to be real.

**s**

[sequence of ints, optional] Shape of the FFT.

Changed in version 2.0: If it is -1, the whole input is used (no padding/trimming).

Deprecated since version 2.0: If *s* is not None, *axes* must not be None either.

Deprecated since version 2.0: *s* must contain only int s, not None values. None values currently mean that the default value for n is used in the corresponding 1-D transform, but this behaviour is deprecated.

#### **axes**

[sequence of ints, optional] Axes over which to compute the FFT. Default: (-2, -1).

Deprecated since version 2.0: If *s* is specified, the corresponding *axes* to be transformed must not be None.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype for the last inverse transform. incompatible with passing in all but the trivial s).

New in version 2.0.0.

#### **Returns**

**out**

[ndarray] The result of the real 2-D FFT.

**See also:**

#### *rfftn*

Compute the N-dimensional discrete Fourier Transform for real input.

### **Notes**

This is really just *rfftn* with different default behavior. For more details see *rfftn*.

### **Examples**

```
>>> import numpy as np
>>> a = np.mgrid[:5, :5][0]
>>> np.fft.rfft2(a)
array([[ 50. +0.j , 0. +0.j , 0. +0.j ],
     [-12.5+17.20477401j, 0. +0.j , 0. +0.j ],
     [-12.5 +4.0614962j , 0. +0.j , 0. +0.j ],
     [-12.5 -4.0614962j , 0. +0.j , 0. +0.j ],
     [-12.5-17.20477401j, 0. +0.j , 0. +0.j ]])
```
### fft.**irfft2**(*a*, *s=None*, *axes=(-2, -1)*, *norm=None*, *out=None*)

Computes the inverse of *rfft2*.

#### **Parameters**

#### **a**

[array_like] The input array

#### **s**

[sequence of ints, optional] Shape of the real output to the inverse FFT.

Changed in version 2.0: If it is -1, the whole input is used (no padding/trimming).

Deprecated since version 2.0: If *s* is not None, *axes* must not be None either.

Deprecated since version 2.0: *s* must contain only int s, not None values. None values currently mean that the default value for n is used in the corresponding 1-D transform, but this behaviour is deprecated.

#### **axes**

[sequence of ints, optional] The axes over which to compute the inverse fft. Default: (-2, -1), the last two axes.

Deprecated since version 2.0: If *s* is specified, the corresponding *axes* to be transformed must not be None.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype for the last transformation.

New in version 2.0.0.

#### **Returns**

**out**

[ndarray] The result of the inverse real 2-D FFT.

**See also:**

### *rfft2*

The forward two-dimensional FFT of real input, of which *irfft2* is the inverse.

*rfft* The one-dimensional FFT for real input.

*irfft*

The inverse of the one-dimensional FFT of real input.

*irfftn*

Compute the inverse of the N-dimensional FFT of real input.

### **Notes**

This is really *irfftn* with different defaults. For more details see *irfftn*.

### **Examples**

```
>>> import numpy as np
>>> a = np.mgrid[:5, :5][0]
>>> A = np.fft.rfft2(a)
>>> np.fft.irfft2(A, s=a.shape)
array([[0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1.],
       [2., 2., 2., 2., 2.],
       [3., 3., 3., 3., 3.],
       [4., 4., 4., 4., 4.]])
```
fft.**rfftn**(*a*, *s=None*, *axes=None*, *norm=None*, *out=None*)

Compute the N-dimensional discrete Fourier Transform for real input.

This function computes the N-dimensional discrete Fourier Transform over any number of axes in an Mdimensional real array by means of the Fast Fourier Transform (FFT). By default, all axes are transformed, with the real transform performed over the last axis, while the remaining transforms are complex.

#### **Parameters**

**a**

[array_like] Input array, taken to be real.

**s**

[sequence of ints, optional] Shape (length along each transformed axis) to use from the input. (s[0] refers to axis 0, s[1] to axis 1, etc.). The final element of *s* corresponds to *n* for rfft(x, n), while for the remaining axes, it corresponds to *n* for fft(x, n). Along any axis, if the given shape is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros.

Changed in version 2.0: If it is -1, the whole input is used (no padding/trimming).

If *s* is not given, the shape of the input along the axes specified by *axes* is used.

Deprecated since version 2.0: If *s* is not None, *axes* must not be None either.

Deprecated since version 2.0: *s* must contain only int s, not None values. None values currently mean that the default value for n is used in the corresponding 1-D transform, but this behaviour is deprecated.

#### **axes**

[sequence of ints, optional] Axes over which to compute the FFT. If not given, the last len(s) axes are used, or all axes if *s* is also not specified.

Deprecated since version 2.0: If *s* is specified, the corresponding *axes* to be transformed must be explicitly specified too.

### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype for all axes (and hence is incompatible with passing in all but the trivial s).

New in version 2.0.0.

### **Returns**

#### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axes indicated by *axes*, or by a combination of *s* and *a*, as explained in the parameters section above. The length of the last axis transformed will be s[-1]//2+1, while the remaining transformed axes will have lengths according to *s*, or unchanged from the input.

#### **Raises**

### **ValueError**

If *s* and *axes* have different length.

#### **IndexError**

If an element of *axes* is larger than than the number of axes of *a*.

#### **See also:**

#### *irfftn*

The inverse of *rfftn*, i.e. the inverse of the n-dimensional FFT of real input.

### *fft*

The one-dimensional FFT, with definitions and conventions used.

#### *rfft*

The one-dimensional FFT of real input.

### *fftn*

The n-dimensional FFT.

#### *rfft2*

The two-dimensional FFT of real input.

#### **Notes**

The transform for real input is performed over the last transformation axis, as by *rfft*, then the transform over the remaining axes is performed as by *fftn*. The order of the output is as for *rfft* for the final transformation axis, and as for *fftn* for the remaining transformation axes.

See *fft* for details, definitions and conventions used.

#### **Examples**

```
>>> import numpy as np
>>> a = np.ones((2, 2, 2))
>>> np.fft.rfftn(a)
array([[[8.+0.j, 0.+0.j], # may vary
        [0.+0.j, 0.+0.j]],
       [[0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j]]])
```

```
>>> np.fft.rfftn(a, axes=(2, 0))
array([[[4.+0.j, 0.+0.j], # may vary
        [4.+0.j, 0.+0.j]],
       [[0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j]]])
```
#### fft.**irfftn**(*a*, *s=None*, *axes=None*, *norm=None*, *out=None*)

Computes the inverse of *rfftn*.

This function computes the inverse of the N-dimensional discrete Fourier Transform for real input over any number of axes in an M-dimensional array by means of the Fast Fourier Transform (FFT). In other words, irfftn(rfftn(a), a.shape) == a to within numerical accuracy. (The a.shape is necessary like len(a) is for *irfft*, and for the same reason.)

The input should be ordered in the same way as is returned by *rfftn*, i.e. as for *irfft* for the final transformation axis, and as for *ifftn* along all the other axes.

#### **Parameters**

#### **a**

[array_like] Input array.

**s**

[sequence of ints, optional] Shape (length of each transformed axis) of the output (s[0] refers to axis 0, s[1] to axis 1, etc.). *s* is also the number of input points used along this axis, except for the last axis, where s[-1]//2+1 points of the input are used. Along any axis, if the shape indicated by *s* is smaller than that of the input, the input is cropped. If it is larger, the input is padded with zeros.

Changed in version 2.0: If it is -1, the whole input is used (no padding/trimming).

If *s* is not given, the shape of the input along the axes specified by axes is used. Except for the last axis which is taken to be 2*(m-1) where m is the length of the input along that axis.

Deprecated since version 2.0: If *s* is not None, *axes* must not be None either.

Deprecated since version 2.0: *s* must contain only int s, not None values. None values currently mean that the default value for n is used in the corresponding 1-D transform, but this behaviour is deprecated.

#### **axes**

[sequence of ints, optional] Axes over which to compute the inverse FFT. If not given, the last *len(s)* axes are used, or all axes if *s* is also not specified. Repeated indices in *axes* means that the inverse transform over that axis is performed multiple times.

Deprecated since version 2.0: If *s* is specified, the corresponding *axes* to be transformed must be explicitly specified too.

### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

### **out**

[ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype for the last transformation.

New in version 2.0.0.

### **Returns**

### **out**

[ndarray] The truncated or zero-padded input, transformed along the axes indicated by *axes*, or by a combination of *s* or *a*, as explained in the parameters section above. The length of each transformed axis is as given by the corresponding element of *s*, or the length of the input in every axis except for the last one if *s* is not given. In the final transformed axis the length of the output when *s* is not given is 2*(m-1) where m is the length of the final transformed axis of the input. To get an odd number of output points in the final axis, *s* must be specified.

### **Raises**

#### **ValueError**

If *s* and *axes* have different length.

### **IndexError**

If an element of *axes* is larger than than the number of axes of *a*.

#### **See also:**

### *rfftn*

The forward n-dimensional FFT of real input, of which *ifftn* is the inverse.

### *fft*

The one-dimensional FFT, with definitions and conventions used.

### *irfft*

The inverse of the one-dimensional FFT of real input.

### *irfft2*

The inverse of the two-dimensional FFT of real input.

### **Notes**

See *fft* for definitions and conventions used.

See *rfft* for definitions and conventions used for real input.

The correct interpretation of the hermitian input depends on the shape of the original data, as given by *s*. This is because each input shape could correspond to either an odd or even length signal. By default, *irfftn* assumes an even output length which puts the last entry at the Nyquist frequency; aliasing with its symmetric counterpart. When performing the final complex to real transform, the last value is thus treated as purely real. To avoid losing information, the correct shape of the real input **must** be given.

### **Examples**

```
>>> import numpy as np
>>> a = np.zeros((3, 2, 2))
>>> a[0, 0, 0] = 3 * 2 * 2
>>> np.fft.irfftn(a)
array([[[1., 1.],
        [1., 1.]],
       [[1., 1.],
        [1., 1.]],
       [[1., 1.],
        [1., 1.]]])
```
### **Hermitian FFTs**

| hfft(a[, n, axis, norm, out]) | Compute the FFT of a signal that has Hermitian symme |
| --- | --- |
|  | try, i.e., a real spectrum. |
| ihfft(a[, n, axis, norm, out]) | Compute the inverse FFT of a signal that has Hermitian |
|  | symmetry. |

### fft.**hfft**(*a*, *n=None*, *axis=-1*, *norm=None*, *out=None*)

Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum.

### **Parameters**

### **a**

[array_like] The input array.

**n**

[int, optional] Length of the transformed axis of the output. For *n* output points, n//2 + 1 input points are necessary. If the input is longer than this, it is cropped. If it is shorter than this, it is padded with zeros. If *n* is not given, it is taken to be 2*(m-1) where m is the length of the input along the axis specified by *axis*.

### **axis**

[int, optional] Axis over which to compute the FFT. If not given, the last axis is used.

### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype.

New in version 2.0.0.

#### **Returns**

#### **out**

[ndarray] The truncated or zero-padded input, transformed along the axis indicated by *axis*, or the last one if *axis* is not specified. The length of the transformed axis is *n*, or, if *n* is not given, 2*m - 2 where m is the length of the transformed axis of the input. To get an odd number of output points, *n* must be specified, for instance as 2*m - 1 in the typical case,

### **Raises**

#### **IndexError**

If *axis* is not a valid axis of *a*.

#### **See also:**

#### *rfft*

Compute the one-dimensional FFT for real input.

#### *ihfft*

The inverse of *hfft*.

#### **Notes**

*hfft*/*ihfft* are a pair analogous to *rfft*/*irfft*, but for the opposite case: here the signal has Hermitian symmetry in the time domain and is real in the frequency domain. So here it's *hfft* for which you must supply the length of the result if it is to be odd.

- even: ihfft(hfft(a, 2*len(a) 2)) == a, within roundoff error,
- odd: ihfft(hfft(a, 2*len(a) 1)) == a, within roundoff error.

The correct interpretation of the hermitian input depends on the length of the original data, as given by *n*. This is because each input shape could correspond to either an odd or even length signal. By default, *hfft* assumes an even output length which puts the last entry at the Nyquist frequency; aliasing with its symmetric counterpart. By Hermitian symmetry, the value is thus treated as purely real. To avoid losing information, the shape of the full signal **must** be given.

#### **Examples**

```
>>> import numpy as np
>>> signal = np.array([1, 2, 3, 4, 3, 2])
>>> np.fft.fft(signal)
array([15.+0.j, -4.+0.j, 0.+0.j, -1.-0.j, 0.+0.j, -4.+0.j]) # may vary
>>> np.fft.hfft(signal[:4]) # Input first half of signal
array([15., -4., 0., -1., 0., -4.])
>>> np.fft.hfft(signal, 6) # Input entire signal and truncate
array([15., -4., 0., -1., 0., -4.])
```

```
>>> signal = np.array([[1, 1.j], [-1.j, 2]])
>>> np.conj(signal.T) - signal # check Hermitian symmetry
array([[ 0.-0.j, -0.+0.j], # may vary
```
(continues on next page)

(continued from previous page)

```
[ 0.+0.j, 0.-0.j]])
>>> freq_spectrum = np.fft.hfft(signal)
>>> freq_spectrum
array([[ 1., 1.],
       [ 2., -2.]])
```
#### fft.**ihfft**(*a*, *n=None*, *axis=-1*, *norm=None*, *out=None*)

Compute the inverse FFT of a signal that has Hermitian symmetry.

#### **Parameters**

**a**

[array_like] Input array.

#### **n**

[int, optional] Length of the inverse FFT, the number of points along transformation axis in the input to use. If *n* is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros. If *n* is not given, the length of the input along the axis specified by *axis* is used.

#### **axis**

[int, optional] Axis over which to compute the inverse FFT. If not given, the last axis is used.

#### **norm**

[{"backward", "ortho", "forward"}, optional] Normalization mode (see *numpy.fft*). Default is "backward". Indicates which direction of the forward/backward pair of transforms is scaled and with what normalization factor.

New in version 1.20.0: The "backward", "forward" values were added.

#### **out**

[complex ndarray, optional] If provided, the result will be placed in this array. It should be of the appropriate shape and dtype.

New in version 2.0.0.

### **Returns**

### **out**

[complex ndarray] The truncated or zero-padded input, transformed along the axis indicated by *axis*, or the last one if *axis* is not specified. The length of the transformed axis is n//2 + 1.

### **See also:**

#### *hfft***,** *irfft*

#### **Notes**

*hfft*/*ihfft* are a pair analogous to *rfft*/*irfft*, but for the opposite case: here the signal has Hermitian symmetry in the time domain and is real in the frequency domain. So here it's *hfft* for which you must supply the length of the result if it is to be odd:

- even: ihfft(hfft(a, 2*len(a) 2)) == a, within roundoff error,
- odd: ihfft(hfft(a, 2*len(a) 1)) == a, within roundoff error.

### **Examples**

```
>>> import numpy as np
>>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
>>> np.fft.ifft(spectrum)
array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j, 3.+0.j, 2.+0.j]) # may vary
>>> np.fft.ihfft(spectrum)
array([ 1.-0.j, 2.-0.j, 3.-0.j, 4.-0.j]) # may vary
```
### **Helper routines**

| fftfreq(n[, d, device]) | Return the Discrete Fourier Transform sample frequen |
| --- | --- |
|  | cies. |
| rfftfreq(n[, d, device]) | Return the Discrete Fourier Transform sample frequen |
|  | cies (for usage with rfft, irfft). |
| fftshift(x[, axes]) | Shift the zero-frequency component to the center of the |
|  | spectrum. |
| ifftshift(x[, axes]) | The inverse of fftshift. |

### fft.**fftfreq**(*n*, *d=1.0*, *device=None*)

Return the Discrete Fourier Transform sample frequencies.

The returned float array *f* contains the frequency bin centers in cycles per unit of the sample spacing (with zero at the start). For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.

Given a window length *n* and a sample spacing *d*:

| f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n) | if n is even |
| --- | --- |
| f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n) | if n is odd |

### **Parameters**

### **n**

[int] Window length.

### **d**

[scalar, optional] Sample spacing (inverse of the sampling rate). Defaults to 1.

### **device**

[str, optional] The device on which to place the created array. Default: None. For Array-API interoperability only, so must be "cpu" if passed.

New in version 2.0.0.

### **Returns**

### **f**

[ndarray] Array of length *n* containing the sample frequencies.

#### **Examples**

```
>>> import numpy as np
>>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
>>> fourier = np.fft.fft(signal)
>>> n = signal.size
>>> timestep = 0.1
>>> freq = np.fft.fftfreq(n, d=timestep)
>>> freq
array([ 0. , 1.25, 2.5 , ..., -3.75, -2.5 , -1.25])
```
#### fft.**rfftfreq**(*n*, *d=1.0*, *device=None*)

Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft).

The returned float array *f* contains the frequency bin centers in cycles per unit of the sample spacing (with zero at the start). For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.

Given a window length *n* and a sample spacing *d*:

f = [0, 1, ..., n/2-1, n/2] / (d*n) **if** n **is** even f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n) **if** n **is** odd

Unlike *fftfreq* (but like scipy.fftpack.rfftfreq) the Nyquist frequency component is considered to be positive.

#### **Parameters**

#### **n**

[int] Window length.

**d**

[scalar, optional] Sample spacing (inverse of the sampling rate). Defaults to 1.

#### **device**

[str, optional] The device on which to place the created array. Default: None. For Array-API interoperability only, so must be "cpu" if passed.

New in version 2.0.0.

#### **Returns**

### **f**

[ndarray] Array of length n//2 + 1 containing the sample frequencies.

### **Examples**

```
>>> import numpy as np
>>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
>>> fourier = np.fft.rfft(signal)
>>> n = signal.size
>>> sample_rate = 100
>>> freq = np.fft.fftfreq(n, d=1./sample_rate)
>>> freq
array([ 0., 10., 20., ..., -30., -20., -10.])
>>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
>>> freq
array([ 0., 10., 20., 30., 40., 50.])
```
#### fft.**fftshift**(*x*, *axes=None*)

Shift the zero-frequency component to the center of the spectrum.

This function swaps half-spaces for all axes listed (defaults to all). Note that y[0] is the Nyquist component only if len(x) is even.

### **Parameters**

**x**

[array_like] Input array.

**axes**

[int or shape tuple, optional] Axes over which to shift. Default is None, which shifts all axes.

### **Returns**

**y**

[ndarray] The shifted array.

### **See also:**

### *ifftshift*

The inverse of *fftshift*.

### **Examples**

```
>>> import numpy as np
>>> freqs = np.fft.fftfreq(10, 0.1)
>>> freqs
array([ 0., 1., 2., ..., -3., -2., -1.])
>>> np.fft.fftshift(freqs)
array([-5., -4., -3., -2., -1., 0., 1., 2., 3., 4.])
```
Shift the zero-frequency component only along the second axis:

```
>>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
>>> freqs
array([[ 0., 1., 2.],
       [ 3., 4., -4.],
       [-3., -2., -1.]])
>>> np.fft.fftshift(freqs, axes=(1,))
array([[ 2., 0., 1.],
       [-4., 3., 4.],
       [-1., -3., -2.]])
```
fft.**ifftshift**(*x*, *axes=None*)

The inverse of *fftshift*. Although identical for even-length *x*, the functions differ by one sample for odd-length *x*.

#### **Parameters**

### **x**

[array_like] Input array.

#### **axes**

[int or shape tuple, optional] Axes over which to calculate. Defaults to None, which shifts all axes.

**Returns**

**y**

- [ndarray] The shifted array.
### **See also:**

### *fftshift*

Shift zero-frequency component to the center of the spectrum.

### **Examples**

```
>>> import numpy as np
>>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
>>> freqs
array([[ 0., 1., 2.],
       [ 3., 4., -4.],
       [-3., -2., -1.]])
>>> np.fft.ifftshift(np.fft.fftshift(freqs))
array([[ 0., 1., 2.],
       [ 3., 4., -4.],
       [-3., -2., -1.]])
```
### **Background information**

Fourier analysis is fundamentally a method for expressing a function as a sum of periodic components, and for recovering the function from those components. When both the function and its Fourier transform are replaced with discretized counterparts, it is called the discrete Fourier transform (DFT). The DFT has become a mainstay of numerical computing in part because of a very fast algorithm for computing it, called the Fast Fourier Transform (FFT), which was known to Gauss (1805) and was brought to light in its current form by Cooley and Tukey [CT]. Press et al. [NR] provide an accessible introduction to Fourier analysis and its applications.

Because the discrete Fourier transform separates its input into components that contribute at discrete frequencies, it has a great number of applications in digital signal processing, e.g., for filtering, and in this context the discretized input to the transform is customarily referred to as a *signal*, which exists in the *time domain*. The output is called a *spectrum* or *transform* and exists in the *frequency domain*.

### **Implementation details**

There are many ways to define the DFT, varying in the sign of the exponent, normalization, etc. In this implementation, the DFT is defined as

$$A_{k}=\sum_{m=0}^{n-1}a_{m}\exp\left\{-2\pi i{\frac{m k}{n}}\right\}\qquad k=0,\ldots,n-1.$$

The DFT is in general defined for complex inputs and outputs, and a single-frequency component at linear frequency *f* is represented by a complex exponential *am* = exp*{*2*πi fm*∆*t}*, where ∆*t* is the sampling interval.

The values in the result follow so-called "standard" order: If A = fft(a, n), then A[0] contains the zero-frequency term (the sum of the signal), which is always purely real for real inputs. Then A[1:n/2] contains the positive-frequency terms, and A[n/2+1:] contains the negative-frequency terms, in order of decreasingly negative frequency. For an even number of input points, A[n/2] represents both positive and negative Nyquist frequency, and is also purely real for real input. For an odd number of input points, A[(n-1)/2] contains the largest positive frequency, while A[(n+1)/2] contains the largest negative frequency. The routine np.fft.fftfreq(n) returns an array giving the frequencies of corresponding elements in the output. The routine np.fft.fftshift(A) shifts transforms and their frequencies to put the zero-frequency components in the middle, and np.fft.ifftshift(A) undoes that shift.

When the input *a* is a time-domain signal and A = fft(a), np.abs(A) is its amplitude spectrum and np. abs(A)**2 is its power spectrum. The phase spectrum is obtained by np.angle(A).

The inverse DFT is defined as

$$a_{m}={\frac{1}{n}}\sum_{k=0}^{n-1}A_{k}\exp\left\{2\pi i{\frac{m k}{n}}\right\}\qquad m=0,\ldots,n-1.$$

It differs from the forward transform by the sign of the exponential argument and the default normalization by 1/*n*.

#### **Type Promotion**

*numpy.fft* promotes float32 and complex64 arrays to float64 and complex128 arrays respectively. For an FFT implementation that does not promote input arrays, see scipy.fftpack.

#### **Normalization**

The argument norm indicates which direction of the pair of direct/inverse transforms is scaled and with what normalization factor. The default normalization ("backward") has the direct (forward) transforms unscaled and the inverse (backward) transforms scaled by 1/*n*. It is possible to obtain unitary transforms by setting the keyword argument norm to "ortho" so that both direct and inverse transforms are scaled by 1/*√ n*. Finally, setting the keyword argument norm to "forward" has the direct transforms scaled by 1/*n* and the inverse transforms unscaled (i.e. exactly opposite to the default "backward"). *None* is an alias of the default option "backward" for backward compatibility.

#### **Real and Hermitian transforms**

When the input is purely real, its transform is Hermitian, i.e., the component at frequency *fk* is the complex conjugate of the component at frequency *−fk*, which means that for real inputs there is no information in the negative frequency components that is not already available from the positive frequency components. The family of *rfft* functions is designed to operate on real inputs, and exploits this symmetry by computing only the positive frequency components, up to and including the Nyquist frequency. Thus, n input points produce n/2+1 complex output points. The inverses of this family assumes the same symmetry of its input, and for an output of n points uses n/2+1 input points.

Correspondingly, when the spectrum is purely real, the signal is Hermitian. The *hfft* family of functions exploits this symmetry by using n/2+1 complex points in the input (time) domain for n real points in the frequency domain.

In higher dimensions, FFTs are used, e.g., for image analysis and filtering. The computational efficiency of the FFT means that it can also be a faster way to compute large convolutions, using the property that a convolution in the time domain is equivalent to a point-by-point multiplication in the frequency domain.

#### **Higher dimensions**

In two dimensions, the DFT is defined as

$$A_{kl}=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}a_{mn}\exp\left\{-2\pi i\left(\frac{mk}{M}+\frac{nl}{N}\right)\right\}\qquad k=0,\ldots,M-1;\quad l=0,\ldots,N-1,$$

which extends in the obvious way to higher dimensions, and the inverses in higher dimensions also extend in the same way.

#### **References**

#### **Examples**

For examples, see the various functions.

### **Linear algebra (numpy.linalg)**

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient low level implementations of standard linear algebra algorithms. Those libraries may be provided by NumPy itself using C versions of a subset of their reference implementations but, when possible, highly optimized libraries that take advantage of specialized processor functionality are preferred. Examples of such libraries are OpenBLAS, MKL (TM), and ATLAS. Because those libraries are multithreaded and processor dependent, environmental variables and external packages such as threadpoolctl may be needed to control the number of threads or specify the processor architecture.

The SciPy library also contains a linalg submodule, and there is overlap in the functionality provided by the SciPy and NumPy submodules. SciPy contains functions not found in *numpy.linalg*, such as functions related to LU decomposition and the Schur decomposition, multiple ways of calculating the pseudoinverse, and matrix transcendentals such as the matrix logarithm. Some functions that exist in both have augmented functionality in scipy.linalg. For example, scipy.linalg.eig can take a second matrix argument for solving generalized eigenvalue problems. Some functions in NumPy, however, have more flexible broadcasting options. For example, *numpy.linalg.solve* can handle "stacked" arrays, while scipy.linalg.solve accepts only a single square array as its first argument.

**Note:** The term *matrix* as it is used on this page indicates a 2d *numpy.array* object, and *not* a *numpy.matrix* object. The latter is no longer recommended, even for linear algebra. See *the matrix object documentation* for more information.

### **The @ operator**

Introduced in NumPy 1.10.0, the @ operator is preferable to other methods when computing the matrix product between 2d arrays. The *numpy.matmul* function implements the @ operator.

### **Matrix and vector products**

| dot(a, b[, out]) | Dot product of two arrays. |
| --- | --- |
| linalg.multi_dot(arrays, *[, out]) | Compute the dot product of two or more arrays in a sin |
|  | gle function call, while automatically selecting the fastest |
|  | evaluation order. |
| vdot(a, b, /) | Return the dot product of two vectors. |
| vecdot(x1, x2, /[, out, casting, order, ...]) | Vector dot product of two arrays. |
| linalg.vecdot(x1, x2, /, *[, axis]) | Computes the vector dot product. |
| inner(a, b, /) | Inner product of two arrays. |
| outer(a, b[, out]) | Compute the outer product of two vectors. |
| matmul(x1, x2, /[, out, casting, order, ...]) | Matrix product of two arrays. |
| linalg.matmul(x1, x2, /) | Computes the matrix product. |
| matvec(x1, x2, /[, out, casting, order, ...]) | Matrix-vector dot product of two arrays. |
| vecmat(x1, x2, /[, out, casting, order, ...]) | Vector-matrix dot product of two arrays. |
| tensordot(a, b[, axes]) | Compute tensor dot product along specified axes. |
| linalg.tensordot(x1, x2, /, *[, axes]) | Compute tensor dot product along specified axes. |
| einsum(subscripts, *operands[, out, dtype, ...]) | Evaluates the Einstein summation convention on the |
|  | operands. |
| einsum_path(subscripts, *operands[, optimize]) | Evaluates the lowest cost contraction order for an einsum |
|  | expression by considering the creation of intermediate ar |
|  | rays. |
| linalg.matrix_power(a, n) | Raise a square matrix to the (integer) power n. |
| kron(a, b) | Kronecker product of two arrays. |
| linalg.cross(x1, x2, /, *[, axis]) | Returns the cross product of 3-element vectors. |

#### numpy.**dot**(*a*, *b*, *out=None*)

Dot product of two arrays. Specifically,

- If both *a* and *b* are 1-D arrays, it is inner product of vectors (without complex conjugation).
- If both *a* and *b* are 2-D arrays, it is matrix multiplication, but using *matmul* or a @ b is preferred.
- If either *a* or *b* is 0-D (scalar), it is equivalent to *multiply* and using numpy.multiply(a, b) or a * b is preferred.
- If *a* is an N-D array and *b* is a 1-D array, it is a sum product over the last axis of *a* and *b*.
- If *a* is an N-D array and *b* is an M-D array (where M>=2), it is a sum product over the last axis of *a* and the second-to-last axis of *b*:

```
dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
```
It uses an optimized BLAS library when possible (see *numpy.linalg*).

#### **Parameters**

**a**

[array_like] First argument.

#### **b**

[array_like] Second argument.

#### **out**

[ndarray, optional] Output argument. This must have the exact kind that would be returned if it was not used. In particular, it must have the right type, must be C-contiguous, and its dtype must be the dtype that would be returned for *dot(a,b)*. This is a performance feature. Therefore, if these conditions are not met, an exception is raised, instead of attempting to be flexible.

#### **Returns**

### **output**

[ndarray] Returns the dot product of *a* and *b*. If *a* and *b* are both scalars or both 1-D arrays then a scalar is returned; otherwise an array is returned. If *out* is given, then it is returned.

#### **Raises**

#### **ValueError**

If the last dimension of *a* is not the same size as the second-to-last dimension of *b*.

#### **See also:**

### *vdot*

Complex-conjugating dot product.

### *vecdot*

Vector dot product of two arrays.

### *tensordot*

Sum products over arbitrary axes.

#### *einsum*

Einstein summation convention.

#### *matmul*

'@' operator as method with out parameter.

#### *linalg.multi_dot*

Chained dot product.

**Examples**

```
>>> import numpy as np
>>> np.dot(3, 4)
12
```
Neither argument is complex-conjugated:

**>>>** np.dot([2j, 3j], [2j, 3j]) (-13+0j)

For 2-D arrays it is the matrix product:

```
>>> a = [[1, 0], [0, 1]]
>>> b = [[4, 1], [2, 2]]
>>> np.dot(a, b)
array([[4, 1],
       [2, 2]])
```

```
>>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
>>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
>>> np.dot(a, b)[2,3,2,1,2,2]
499128
>>> sum(a[2,3,2,:] * b[1,2,:,2])
499128
```
linalg.**multi_dot**(*arrays*, ***, *out=None*)

Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.

*multi_dot* chains *numpy.dot* and uses optimal parenthesization of the matrices [1] [2]. Depending on the shapes of the matrices, this can speed up the multiplication a lot.

If the first argument is 1-D it is treated as a row vector. If the last argument is 1-D it is treated as a column vector. The other arguments must be 2-D.

Think of *multi_dot* as:

**def** multi_dot(arrays): **return** functools.reduce(np.dot, arrays)

#### **Parameters**

### **arrays**

[sequence of array_like] If the first argument is 1-D it is treated as row vector. If the last argument is 1-D it is treated as column vector. The other arguments must be 2-D.

**out**

[ndarray, optional] Output argument. This must have the exact kind that would be returned if it was not used. In particular, it must have the right type, must be C-contiguous, and its dtype must be the dtype that would be returned for *dot(a, b)*. This is a performance feature. Therefore, if these conditions are not met, an exception is raised, instead of attempting to be flexible.

#### **Returns**

#### **output**

[ndarray] Returns the dot product of the supplied arrays.

**See also:**

```
numpy.dot
```
dot multiplication with two arguments.

### **Notes**

The cost for a matrix multiplication can be calculated with the following function:

**def** cost(A, B): **return** A.shape[0] * A.shape[1] * B.shape[1]

Assume we have three matrices *A*10*x*100*, B*100*x*5*, C*5*x*50.

The costs for the two different parenthesizations are as follows:

cost((AB)C) = 10*100*5 + 10*5*50 = 5000 + 2500 = 7500 cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000

### **References**

[1], [2]

### **Examples**

*multi_dot* allows you to write:

```
>>> import numpy as np
>>> from numpy.linalg import multi_dot
>>> # Prepare some data
>>> A = np.random.random((10000, 100))
>>> B = np.random.random((100, 1000))
>>> C = np.random.random((1000, 5))
>>> D = np.random.random((5, 333))
>>> # the actual dot multiplication
>>> _ = multi_dot([A, B, C, D])
```
instead of:

```
>>> _ = np.dot(np.dot(np.dot(A, B), C), D)
>>> # or
>>> _ = A.dot(B).dot(C).dot(D)
```
numpy.**vdot**(*a*, *b*, */* )

Return the dot product of two vectors.

The *vdot* function handles complex numbers differently than *dot*: if the first argument is complex, it is replaced by its complex conjugate in the dot product calculation. *vdot* also handles multidimensional arrays differently than *dot*: it does not perform a matrix product, but flattens the arguments to 1-D arrays before taking a vector dot product.

Consequently, when the arguments are 2-D arrays of the same shape, this function effectively returns their Frobenius inner product (also known as the *trace inner product* or the *standard inner product* on a vector space of matrices).

### **Parameters**

### **a**

[array_like] If *a* is complex the complex conjugate is taken before calculation of the dot product.

#### **b**

[array_like] Second argument to the dot product.

### **Returns**

### **output**

[ndarray] Dot product of *a* and *b*. Can be an int, float, or complex depending on the types of *a* and *b*.

### **See also:**

*dot*

Return the dot product without using the complex conjugate of the first argument.

#### **Examples**

```
>>> import numpy as np
>>> a = np.array([1+2j,3+4j])
>>> b = np.array([5+6j,7+8j])
>>> np.vdot(a, b)
(70-8j)
>>> np.vdot(b, a)
(70+8j)
```
Note that higher-dimensional arrays are flattened!

```
>>> a = np.array([[1, 4], [5, 6]])
>>> b = np.array([[4, 1], [2, 2]])
>>> np.vdot(a, b)
30
>>> np.vdot(b, a)
30
>>> 1*4 + 4*1 + 5*2 + 6*2
30
```
numpy.**vecdot**(*x1*, *x2*, */* , *out=None*, ***, *casting='same_kind'*, *order='K'*, *dtype=None*, *subok=True*[, *signature*, *axes*, *axis*]) **= <ufunc 'vecdot'>**

Vector dot product of two arrays.

Let **a** be a vector in *x1* and **b** be a corresponding vector in *x2*. The dot product is defined as:

$$\mathbf{a}\cdot\mathbf{b}=\sum_{i=0}^{n-1}{\overline{{a_{i}}}}b_{i}$$

where the sum is over the last dimension (unless *axis* is specified) and where *ai* denotes the complex conjugate if *ai* is complex and the identity otherwise.

New in version 2.0.0.

#### **Parameters**

**x1, x2**

[array_like] Input arrays, scalars not allowed.

#### **out**

[ndarray, optional] A location into which the result is stored. If provided, it must have the broadcasted shape of *x1* and *x2* with the last axis removed. If not provided or None, a freshlyallocated array is used.

#### ****kwargs**

For other keyword-only arguments, see the *ufunc docs*.

### **Returns**

**y**

[ndarray] The vector dot product of the inputs. This is a scalar only when both x1, x2 are 1-d vectors.

#### **Raises**

#### **ValueError**

If the last dimension of *x1* is not the same size as the last dimension of *x2*.

If a scalar value is passed in.

#### **See also:**

#### *vdot*

same but flattens arguments first

### *matmul*

Matrix-matrix product.

#### *vecmat*

Vector-matrix product.

### *matvec*

Matrix-vector product.

#### *einsum*

Einstein summation convention.

#### **Examples**

**>>> import numpy as np**

Get the projected size along a given normal for an array of vectors.

```
>>> v = np.array([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]])
>>> n = np.array([0., 0.6, 0.8])
>>> np.vecdot(v, n)
array([ 3., 8., 10.])
```
#### linalg.**vecdot**(*x1*, *x2*, */* , ***, *axis=-1*)

Computes the vector dot product.

This function is restricted to arguments compatible with the Array API, contrary to *numpy.vecdot*.

Let **a** be a vector in x1 and **b** be a corresponding vector in x2. The dot product is defined as:

$$\mathbf{a}\cdot\mathbf{b}=\sum_{i=0}^{n-1}{\overline{{a_{i}}}}b_{i}$$

over the dimension specified by axis and where *ai* denotes the complex conjugate if *ai* is complex and the identity otherwise.

#### **Parameters**

**x1**

[array_like] First input array.

#### **x2**

[array_like] Second input array.

### **axis**

[int, optional] Axis over which to compute the dot product. Default: -1.

### **Returns**

**output**

[ndarray] The vector dot product of the input.

#### **See also:**

*numpy.vecdot*

#### **Examples**

Get the projected size along a given normal for an array of vectors.

```
>>> v = np.array([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]])
>>> n = np.array([0., 0.6, 0.8])
>>> np.linalg.vecdot(v, n)
array([ 3., 8., 10.])
```
#### numpy.**inner**(*a*, *b*, */*)

Inner product of two arrays.

Ordinary inner product of vectors for 1-D arrays (without complex conjugation), in higher dimensions a sum product over the last axes.

#### **Parameters**

### **a, b**

[array_like] If *a* and *b* are nonscalar, their last dimensions must match.

#### **Returns**

### **out**

[ndarray] If *a* and *b* are both scalars or both 1-D arrays then a scalar is returned; otherwise an array is returned. out.shape = (*a.shape[:-1], *b.shape[:-1])

#### **Raises**

```
ValueError
```
If both *a* and *b* are nonscalar and their last dimensions have different sizes.

### **See also:**

#### *tensordot*

Sum products over arbitrary axes.

### *dot*

Generalised matrix product, using second last dimension of *b*.

#### *vecdot*

Vector dot product of two arrays.

#### *einsum*

Einstein summation convention.

### **Notes**

For vectors (1-D arrays) it computes the ordinary inner-product:

```
np.inner(a, b) = sum(a[:]*b[:])
```

```
More generally, if ndim(a) = r > 0 and ndim(b) = s > 0:
```

```
np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))
```
or explicitly:

```
np.inner(a, b)[i0,...,ir-2,j0,...,js-2]
     = sum(a[i0,...,ir-2,:]*b[j0,...,js-2,:])
```
In addition *a* or *b* may be scalars, in which case:

np.inner(a,b) = a*b

### **Examples**

Ordinary inner product for vectors:

```
>>> import numpy as np
>>> a = np.array([1,2,3])
>>> b = np.array([0,1,0])
>>> np.inner(a, b)
2
```
Some multidimensional examples:

```
>>> a = np.arange(24).reshape((2,3,4))
>>> b = np.arange(4)
>>> c = np.inner(a, b)
>>> c.shape
(2, 3)
>>> c
array([[ 14, 38, 62],
       [ 86, 110, 134]])
```

```
>>> a = np.arange(2).reshape((1,1,2))
>>> b = np.arange(6).reshape((3,2))
>>> c = np.inner(a, b)
>>> c.shape
(1, 1, 3)
>>> c
array([[[1, 3, 5]]])
```
An example where *b* is a scalar:

```
>>> np.inner(np.eye(2), 7)
array([[7., 0.],
       [0., 7.]])
```
numpy.**outer**(*a*, *b*, *out=None*)

Compute the outer product of two vectors.

Given two vectors *a* and *b* of length M and N, respectively, the outer product [1] is:

```
[[a_0*b_0 a_0*b_1 ... a_0*b_{N-1} ]
[a_1*b_0 .
[ ... .
[a_{M-1}*b_0 a_{M-1}*b_{N-1} ]]
```
#### **Parameters**

### **a**

[(M,) array_like] First input vector. Input is flattened if not already 1-dimensional.

**b**

[(N,) array_like] Second input vector. Input is flattened if not already 1-dimensional.

**out**

[(M, N) ndarray, optional] A location where the result is stored

#### **Returns**

**out**

```
[(M, N) ndarray] out[i, j] = a[i] * b[j]
```
#### **See also:**

*inner*

#### *einsum*

einsum('i,j->ij', a.ravel(), b.ravel()) is the equivalent.

#### *ufunc.outer*

A generalization to dimensions other than 1D and other operations. np.multiply.outer(a. ravel(), b.ravel()) is the equivalent.

#### *linalg.outer*

An Array API compatible variation of np.outer, which accepts 1-dimensional inputs only.

#### *tensordot*

np.tensordot(a.ravel(), b.ravel(), axes=((), ())) is the equivalent.

#### **References**

[1]

### **Examples**

Make a (*very* coarse) grid for computing a Mandelbrot set:

```
>>> import numpy as np
>>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
>>> rl
array([[-2., -1., 0., 1., 2.],
      [-2., -1., 0., 1., 2.],
      [-2., -1., 0., 1., 2.],
      [-2., -1., 0., 1., 2.],
      [-2., -1., 0., 1., 2.]])
>>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
>>> im
array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
      [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
      [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
      [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
      [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
>>> grid = rl + im
>>> grid
array([[-2.+2.j, -1.+2.j, 0.+2.j, 1.+2.j, 2.+2.j],
      [-2.+1.j, -1.+1.j, 0.+1.j, 1.+1.j, 2.+1.j],
      [-2.+0.j, -1.+0.j, 0.+0.j, 1.+0.j, 2.+0.j],
      [-2.-1.j, -1.-1.j, 0.-1.j, 1.-1.j, 2.-1.j],
      [-2.-2.j, -1.-2.j, 0.-2.j, 1.-2.j, 2.-2.j]])
```
An example using a "vector" of letters:

```
>>> x = np.array(['a', 'b', 'c'], dtype=object)
>>> np.outer(x, [1, 2, 3])
array([['a', 'aa', 'aaa'],
       ['b', 'bb', 'bbb'],
       ['c', 'cc', 'ccc']], dtype=object)
```
numpy.**matmul**(*x1*, *x2*, */* , *out=None*, ***, *casting='same_kind'*, *order='K'*, *dtype=None*, *subok=True*[, *signature*, *axes*, *axis*]) **= <ufunc 'matmul'>**

Matrix product of two arrays.

### **Parameters**

### **x1, x2**

[array_like] Input arrays, scalars not allowed.

### **out**

[ndarray, optional] A location into which the result is stored. If provided, it must have a shape that matches the signature *(n,k),(k,m)->(n,m)*. If not provided or None, a freshly-allocated array is returned.

### ****kwargs**

For other keyword-only arguments, see the *ufunc docs*.

### **Returns**

### **y**

[ndarray] The matrix product of the inputs. This is a scalar only when both x1, x2 are 1-d vectors.

### **Raises**

### **ValueError**

If the last dimension of *x1* is not the same size as the second-to-last dimension of *x2*.

If a scalar value is passed in.

### **See also:**

#### *vecdot*

Complex-conjugating dot product for stacks of vectors.

### *matvec*

Matrix-vector product for stacks of matrices and vectors.

### *vecmat*

Vector-matrix product for stacks of vectors and matrices.

#### *tensordot*

Sum products over arbitrary axes.

### *einsum*

Einstein summation convention.

#### *dot*

alternative matrix product with different broadcasting rules.

#### **Notes**

The behavior depends on the arguments in the following way.

- If both arguments are 2-D they are multiplied like conventional matrices.
- If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
- If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed. (For stacks of vectors, use vecmat.)
- If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed. (For stacks of vectors, use matvec.)

matmul differs from dot in two important ways:

- Multiplication by scalars is not allowed, use * instead.
- Stacks of matrices are broadcast together as if the matrices were elements, respecting the signature (n,k), (k,m)->(n,m):

```
>>> a = np.ones([9, 5, 7, 4])
>>> c = np.ones([9, 5, 4, 3])
>>> np.dot(a, c).shape
(9, 5, 7, 9, 5, 3)
>>> np.matmul(a, c).shape
(9, 5, 7, 3)
>>> # n is 7, k is 4, m is 3
```
The matmul function implements the semantics of the @ operator introduced in Python 3.5 following **PEP 465**.

It uses an optimized BLAS library when possible (see *numpy.linalg*).

#### **Examples**

For 2-D arrays it is the matrix product:

```
>>> import numpy as np
```

```
>>> a = np.array([[1, 0],
... [0, 1]])
>>> b = np.array([[4, 1],
... [2, 2]])
>>> np.matmul(a, b)
array([[4, 1],
     [2, 2]])
```
For 2-D mixed with 1-D, the result is the usual.

```
>>> a = np.array([[1, 0],
... [0, 1]])
>>> b = np.array([1, 2])
>>> np.matmul(a, b)
array([1, 2])
>>> np.matmul(b, a)
array([1, 2])
```
Broadcasting is conventional for stacks of arrays

```
>>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
>>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))
>>> np.matmul(a,b).shape
(2, 2, 2)
>>> np.matmul(a, b)[0, 1, 1]
98
>>> sum(a[0, 1, :] * b[0 , :, 1])
98
```
Vector, vector returns the scalar inner product, but neither argument is complex-conjugated:

```
>>> np.matmul([2j, 3j], [2j, 3j])
(-13+0j)
```
Scalar multiplication raises an error.

```
>>> np.matmul([1,2], 3)
Traceback (most recent call last):
...
ValueError: matmul: Input operand 1 does not have enough dimensions ...
```
The @ operator can be used as a shorthand for np.matmul on ndarrays.

```
>>> x1 = np.array([2j, 3j])
>>> x2 = np.array([2j, 3j])
>>> x1 @ x2
(-13+0j)
```
linalg.**matmul**(*x1*, *x2*, */*)

Computes the matrix product.

This function is Array API compatible, contrary to *numpy.matmul*.

#### **Parameters**

**x1**

**x2**

[array_like] The second input array.

[array_like] The first input array.

#### **Returns**

#### **out**

[ndarray] The matrix product of the inputs. This is a scalar only when both x1, x2 are 1-d vectors.

### **Raises**

#### **ValueError**

If the last dimension of x1 is not the same size as the second-to-last dimension of x2.

If a scalar value is passed in.

#### **See also:**

*numpy.matmul*

### **Examples**

For 2-D arrays it is the matrix product:

```
>>> a = np.array([[1, 0],
... [0, 1]])
>>> b = np.array([[4, 1],
... [2, 2]])
>>> np.linalg.matmul(a, b)
array([[4, 1],
      [2, 2]])
```
For 2-D mixed with 1-D, the result is the usual.

```
>>> a = np.array([[1, 0],
... [0, 1]])
>>> b = np.array([1, 2])
>>> np.linalg.matmul(a, b)
array([1, 2])
>>> np.linalg.matmul(b, a)
array([1, 2])
```
Broadcasting is conventional for stacks of arrays

```
>>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
>>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))
>>> np.linalg.matmul(a,b).shape
(2, 2, 2)
>>> np.linalg.matmul(a, b)[0, 1, 1]
98
>>> sum(a[0, 1, :] * b[0 , :, 1])
98
```
Vector, vector returns the scalar inner product, but neither argument is complex-conjugated:

```
>>> np.linalg.matmul([2j, 3j], [2j, 3j])
(-13+0j)
```
Scalar multiplication raises an error.

```
>>> np.linalg.matmul([1,2], 3)
Traceback (most recent call last):
...
ValueError: matmul: Input operand 1 does not have enough dimensions ...
```
numpy.**matvec**(*x1*, *x2*, */* , *out=None*, ***, *casting='same_kind'*, *order='K'*, *dtype=None*, *subok=True*[, *signature*, *axes*, *axis*]) **= <ufunc 'matvec'>**

Matrix-vector dot product of two arrays.

Given a matrix (or stack of matrices) **A** in x1 and a vector (or stack of vectors) **v** in x2, the matrix-vector product is defined as:

$$\mathbf{A}\cdot\mathbf{b}=\sum_{j=0}^{n-1}A_{i j}v_{j}$$

where the sum is over the last dimensions in x1 and x2 (unless axes is specified). (For a matrix-vector product with the vector conjugated, use np.vecmat(x2, x1.mT).)

New in version 2.2.0.

### **Parameters**

### **x1, x2**

[array_like] Input arrays, scalars not allowed.

#### **out**

[ndarray, optional] A location into which the result is stored. If provided, it must have the broadcasted shape of x1 and x2 with the summation axis removed. If not provided or None, a freshly-allocated array is used.

#### ****kwargs**

For other keyword-only arguments, see the *ufunc docs*.

#### **Returns**

**y**

[ndarray] The matrix-vector product of the inputs.

#### **Raises**

### **ValueError**

If the last dimensions of x1 and x2 are not the same size.

If a scalar value is passed in.

### **See also:**

### *vecdot*

Vector-vector product.

#### *vecmat*

Vector-matrix product.

### *matmul*

Matrix-matrix product.

#### *einsum*

Einstein summation convention.

### **Examples**

Rotate a set of vectors from Y to X along Z.

```
>>> a = np.array([[0., 1., 0.],
... [-1., 0., 0.],
... [0., 0., 1.]])
>>> v = np.array([[1., 0., 0.],
... [0., 1., 0.],
... [0., 0., 1.],
... [0., 6., 8.]])
>>> np.matvec(a, v)
array([[ 0., -1., 0.],
     [ 1., 0., 0.],
     [ 0., 0., 1.],
     [ 6., 0., 8.]])
```
numpy.**vecmat**(*x1*, *x2*, */* , *out=None*, ***, *casting='same_kind'*, *order='K'*, *dtype=None*, *subok=True*[, *signature*, *axes*, *axis*]) **= <ufunc 'vecmat'>**

Vector-matrix dot product of two arrays.

Given a vector (or stack of vector) **v** in x1 and a matrix (or stack of matrices) **A** in x2, the vector-matrix product is defined as:

$$\mathbf{b}\cdot\mathbf{A}=\sum_{i=0}^{n-1}{\overline{{v_{i}}}}A_{i j}$$

where the sum is over the last dimension of x1 and the one-but-last dimensions in x2 (unless *axes* is specified) and where *vi* denotes the complex conjugate if *v* is complex and the identity otherwise. (For a non-conjugated vector-matrix product, use np.matvec(x2.mT, x1).)

New in version 2.2.0.

#### **Parameters**

#### **x1, x2**

[array_like] Input arrays, scalars not allowed.

#### **out**

[ndarray, optional] A location into which the result is stored. If provided, it must have the broadcasted shape of x1 and x2 with the summation axis removed. If not provided or None, a freshly-allocated array is used.

### ****kwargs**

For other keyword-only arguments, see the *ufunc docs*.

#### **Returns**

**y**

[ndarray] The vector-matrix product of the inputs.

#### **Raises**

#### **ValueError**

If the last dimensions of x1 and the one-but-last dimension of x2 are not the same size.

If a scalar value is passed in.

#### **See also:**

#### *vecdot*

Vector-vector product.

### *matvec*

Matrix-vector product.

# *matmul* Matrix-matrix product.

*einsum*

Einstein summation convention.

### **Examples**

Project a vector along X and Y.

```
>>> v = np.array([0., 4., 2.])
>>> a = np.array([[1., 0., 0.],
... [0., 1., 0.],
... [0., 0., 0.]])
>>> np.vecmat(v, a)
array([ 0., 4., 0.])
```
### numpy.**tensordot**(*a*, *b*, *axes=2*)

Compute tensor dot product along specified axes.

Given two tensors, *a* and *b*, and an array_like object containing two array_like objects, (a_axes, b_axes), sum the products of *a*'s and *b*'s elements (components) over the axes specified by a_axes and b_axes. The third argument can be a single non-negative integer_like scalar, N; if it is such, then the last N dimensions of *a* and the first N dimensions of *b* are summed over.

### **Parameters**

### **a, b**

[array_like] Tensors to "dot".

### **axes**

[int or (2,) array_like]

- integer_like If an int N, sum over the last N axes of *a* and the first N axes of *b* in order. The sizes of the corresponding axes must match.
- (2,) array_like Or, a list of axes to be summed over, first sequence applying to *a*, second to *b*. Both elements array_like must be of the same length.

### **Returns**

### **output**

[ndarray] The tensor dot product of the input.

### **See also:**

#### *dot***,** *einsum*

### **Notes**

**Three common use cases are:**

- axes = 0 : tensor product *a ⊗ b*
- axes = 1 : tensor dot product *a · b*
- axes = 2 : (default) tensor double contraction *a* : *b*

When *axes* is integer_like, the sequence of axes for evaluation will be: from the -Nth axis to the -1th axis in *a*, and from the 0th axis to (N-1)th axis in *b*. For example, axes = 2 is the equal to axes = [[-2, -1], [0, 1]]. When N-1 is smaller than 0, or when -N is larger than -1, the element of *a* and *b* are defined as the *axes*.

When there is more than one axis to sum over - and they are not the last (first) axes of *a* (*b*) - the argument *axes* should consist of two sequences of the same length, with the first axis to sum over given first in both sequences, the second axis second, and so forth. The calculation can be referred to numpy.einsum.

The shape of the result consists of the non-contracted axes of the first tensor, followed by the non-contracted axes of the second.

#### **Examples**

An example on integer_like:

```
>>> a_0 = np.array([[1, 2], [3, 4]])
>>> b_0 = np.array([[5, 6], [7, 8]])
>>> c_0 = np.tensordot(a_0, b_0, axes=0)
>>> c_0.shape
(2, 2, 2, 2)
>>> c_0
array([[[[ 5, 6],
         [ 7, 8]],
        [[10, 12],
         [14, 16]]],
       [[[15, 18],
         [21, 24]],
        [[20, 24],
         [28, 32]]]])
```
An example on array_like:

```
>>> a = np.arange(60.).reshape(3,4,5)
>>> b = np.arange(24.).reshape(4,3,2)
>>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
>>> c.shape
(5, 2)
>>> c
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
```
A slower but equivalent way of computing the same…

```
>>> d = np.zeros((5,2))
>>> for i in range(5):
... for j in range(2):
... for k in range(3):
... for n in range(4):
... d[i,j] += a[k,n,i] * b[n,k,j]
>>> c == d
array([[ True, True],
      [ True, True],
      [ True, True],
      [ True, True],
      [ True, True]])
```
An extended example taking advantage of the overloading of + and *:

```
>>> a = np.array(range(1, 9))
>>> a.shape = (2, 2, 2)
>>> A = np.array(('a', 'b', 'c', 'd'), dtype=object)
>>> A.shape = (2, 2)
>>> a; A
array([[[1, 2],
        [3, 4]],
       [[5, 6],
        [7, 8]]])
array([['a', 'b'],
       ['c', 'd']], dtype=object)
```
**>>>** np.tensordot(a, A) *# third argument default is 2 for double-contraction* array(['abbcccdddd', 'aaaaabbbbbbcccccccdddddddd'], dtype=object)

```
>>> np.tensordot(a, A, 1)
array([[['acc', 'bdd'],
        ['aaacccc', 'bbbdddd']],
       [['aaaaacccccc', 'bbbbbdddddd'],
        ['aaaaaaacccccccc', 'bbbbbbbdddddddd']]], dtype=object)
```

```
>>> np.tensordot(a, A, 0) # tensor product (result too long to incl.)
array([[[[['a', 'b'],
          ['c', 'd']],
          ...
```

```
>>> np.tensordot(a, A, (0, 1))
array([[['abbbbb', 'cddddd'],
        ['aabbbbbb', 'ccdddddd']],
       [['aaabbbbbbb', 'cccddddddd'],
        ['aaaabbbbbbbb', 'ccccdddddddd']]], dtype=object)
```

```
>>> np.tensordot(a, A, (2, 1))
array([[['abb', 'cdd'],
        ['aaabbbb', 'cccdddd']],
       [['aaaaabbbbbb', 'cccccdddddd'],
        ['aaaaaaabbbbbbbb', 'cccccccdddddddd']]], dtype=object)
```

```
>>> np.tensordot(a, A, ((0, 1), (0, 1)))
array(['abbbcccccddddddd', 'aabbbbccccccdddddddd'], dtype=object)
```

```
>>> np.tensordot(a, A, ((2, 1), (1, 0)))
array(['acccbbdddd', 'aaaaacccccccbbbbbbdddddddd'], dtype=object)
```
### linalg.**tensordot**(*x1*, *x2*, */* , ***, *axes=2*)

Compute tensor dot product along specified axes.

Given two tensors, *a* and *b*, and an array_like object containing two array_like objects, (a_axes, b_axes), sum the products of *a*'s and *b*'s elements (components) over the axes specified by a_axes and b_axes. The third argument can be a single non-negative integer_like scalar, N; if it is such, then the last N dimensions of *a* and the first N dimensions of *b* are summed over.

#### **Parameters**

**a, b**

[array_like] Tensors to "dot".

#### **axes**

[int or (2,) array_like]

- integer_like If an int N, sum over the last N axes of *a* and the first N axes of *b* in order. The sizes of the corresponding axes must match.
- (2,) array_like Or, a list of axes to be summed over, first sequence applying to *a*, second to *b*. Both elements array_like must be of the same length.

#### **Returns**

**output**

[ndarray] The tensor dot product of the input.

**See also:**

*dot***,** *einsum*

#### **Notes**

#### **Three common use cases are:**

- axes = 0 : tensor product *a ⊗ b*
- axes = 1 : tensor dot product *a · b*
- axes = 2 : (default) tensor double contraction *a* : *b*

When *axes* is integer_like, the sequence of axes for evaluation will be: from the -Nth axis to the -1th axis in *a*, and from the 0th axis to (N-1)th axis in *b*. For example, axes = 2 is the equal to axes = [[-2, -1], [0, 1]]. When N-1 is smaller than 0, or when -N is larger than -1, the element of *a* and *b* are defined as the *axes*.

When there is more than one axis to sum over - and they are not the last (first) axes of *a* (*b*) - the argument *axes* should consist of two sequences of the same length, with the first axis to sum over given first in both sequences, the second axis second, and so forth. The calculation can be referred to numpy.einsum.

The shape of the result consists of the non-contracted axes of the first tensor, followed by the non-contracted axes of the second.

### **Examples**

An example on integer_like:

```
>>> a_0 = np.array([[1, 2], [3, 4]])
>>> b_0 = np.array([[5, 6], [7, 8]])
>>> c_0 = np.tensordot(a_0, b_0, axes=0)
>>> c_0.shape
(2, 2, 2, 2)
>>> c_0
array([[[[ 5, 6],
         [ 7, 8]],
        [[10, 12],
         [14, 16]]],
       [[[15, 18],
         [21, 24]],
        [[20, 24],
         [28, 32]]]])
```
An example on array_like:

```
>>> a = np.arange(60.).reshape(3,4,5)
>>> b = np.arange(24.).reshape(4,3,2)
>>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
>>> c.shape
(5, 2)
>>> c
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
```
A slower but equivalent way of computing the same…

```
>>> d = np.zeros((5,2))
>>> for i in range(5):
... for j in range(2):
... for k in range(3):
... for n in range(4):
... d[i,j] += a[k,n,i] * b[n,k,j]
>>> c == d
array([[ True, True],
      [ True, True],
      [ True, True],
      [ True, True],
      [ True, True]])
```
An extended example taking advantage of the overloading of + and *:

```
>>> a = np.array(range(1, 9))
>>> a.shape = (2, 2, 2)
>>> A = np.array(('a', 'b', 'c', 'd'), dtype=object)
>>> A.shape = (2, 2)
>>> a; A
array([[[1, 2],
        [3, 4]],
       [[5, 6],
```
(continues on next page)

(continued from previous page)

| [7, 8]]]) |
| --- |
| array([['a', 'b'], |
| ['c', 'd']], dtype=object) |

**>>>** np.tensordot(a, A) *# third argument default is 2 for double-contraction* array(['abbcccdddd', 'aaaaabbbbbbcccccccdddddddd'], dtype=object)

```
>>> np.tensordot(a, A, 1)
array([[['acc', 'bdd'],
        ['aaacccc', 'bbbdddd']],
       [['aaaaacccccc', 'bbbbbdddddd'],
        ['aaaaaaacccccccc', 'bbbbbbbdddddddd']]], dtype=object)
```

```
>>> np.tensordot(a, A, 0) # tensor product (result too long to incl.)
array([[[[['a', 'b'],
          ['c', 'd']],
          ...
```

```
>>> np.tensordot(a, A, (0, 1))
array([[['abbbbb', 'cddddd'],
        ['aabbbbbb', 'ccdddddd']],
       [['aaabbbbbbb', 'cccddddddd'],
        ['aaaabbbbbbbb', 'ccccdddddddd']]], dtype=object)
```

```
>>> np.tensordot(a, A, (2, 1))
array([[['abb', 'cdd'],
        ['aaabbbb', 'cccdddd']],
       [['aaaaabbbbbb', 'cccccdddddd'],
        ['aaaaaaabbbbbbbb', 'cccccccdddddddd']]], dtype=object)
```
**>>>** np.tensordot(a, A, ((0, 1), (0, 1))) array(['abbbcccccddddddd', 'aabbbbccccccdddddddd'], dtype=object)

**>>>** np.tensordot(a, A, ((2, 1), (1, 0))) array(['acccbbdddd', 'aaaaacccccccbbbbbbdddddddd'], dtype=object)

numpy.**einsum**(*subscripts*, **operands*, *out=None*, *dtype=None*, *order='K'*, *casting='safe'*, *optimize=False*)

Evaluates the Einstein summation convention on the operands.

Using the Einstein summation convention, many common multi-dimensional, linear algebraic array operations can be represented in a simple fashion. In *implicit* mode *einsum* computes these values.

In *explicit* mode, *einsum* provides further flexibility to compute other array operations that might not be considered classical Einstein summation operations, by disabling, or forcing summation over specified subscript labels.

See the notes and examples for clarification.

#### **Parameters**

#### **subscripts**

[str] Specifies the subscripts for summation as comma separated list of subscript labels. An implicit (classical Einstein summation) calculation is performed unless the explicit indicator '->' is included as well as subscript labels of the precise output form.

#### **operands**

[list of array_like] These are the arrays for the operation.

### **out**

[ndarray, optional] If provided, the calculation is done into this array.

#### **dtype**

[{data-type, None}, optional] If provided, forces the calculation to use the data type specified. Note that you may have to also give a more liberal *casting* parameter to allow the conversions. Default is None.

### **order**

[{'C', 'F', 'A', 'K'}, optional] Controls the memory layout of the output. 'C' means it should be C contiguous. 'F' means it should be Fortran contiguous, 'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise. 'K' means it should be as close to the layout as the inputs as is possible, including arbitrarily permuted axes. Default is 'K'.

#### **casting**

[{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional] Controls what kind of data casting may occur. Setting this to 'unsafe' is not recommended, as it can adversely affect accumulations.

- 'no' means the data types should not be cast at all.
- 'equiv' means only byte-order changes are allowed.
- 'safe' means only casts which can preserve values are allowed.
- 'same_kind' means only safe casts or casts within a kind, like float64 to float32, are allowed.
- 'unsafe' means any data conversions may be done.

Default is 'safe'.

### **optimize**

[{False, True, 'greedy', 'optimal'}, optional] Controls if intermediate optimization should occur. No optimization will occur if False and True will default to the 'greedy' algorithm. Also accepts an explicit contraction list from the np.einsum_path function. See np. einsum_path for more details. Defaults to False.

### **Returns**

### **output**

[ndarray] The calculation based on the Einstein summation convention.

#### **See also:**

#### *einsum_path***,** *dot***,** *inner***,** *outer***,** *tensordot***,** *linalg.multi_dot*

*einsum*

Similar verbose interface is provided by the einops package to cover additional operations: transpose, reshape/flatten, repeat/tile, squeeze/unsqueeze and reductions. The opt_einsum optimizes contraction order for einsum-like expressions in backend-agnostic manner.

### **Notes**

The Einstein summation convention can be used to compute many multi-dimensional, linear algebraic array operations. *einsum* provides a succinct way of representing these.

A non-exhaustive list of these operations, which can be computed by *einsum*, is shown below along with examples:

- Trace of an array, *numpy.trace*.
- Return a diagonal, *numpy.diag*.
- Array axis summations, *numpy.sum*.
- Transpositions and permutations, *numpy.transpose*.
- **Matrix multiplication and dot product,** *numpy.matmul numpy.dot*.
- **Vector inner and outer products,** *numpy.inner numpy.outer*.
- **Broadcasting, element-wise and scalar multiplication,** *numpy.multiply*.
- Tensor contractions, *numpy.tensordot*.
- **Chained array operations, in efficient calculation order,** *numpy.einsum_path*.

The subscripts string is a comma-separated list of subscript labels, where each label refers to a dimension of the corresponding operand. Whenever a label is repeated it is summed, so np.einsum('i,i', a, b) is equivalent to *np.inner(a,b)*. If a label appears only once, it is not summed, so np.einsum('i', a) produces a view of a with no changes. A further example np.einsum('ij,jk', a, b) describes traditional matrix multiplication and is equivalent to *np.matmul(a,b)*. Repeated subscript labels in one operand take the diagonal. For example, np.einsum('ii', a) is equivalent to *np.trace(a)*.

In *implicit mode*, the chosen subscripts are important since the axes of the output are reordered alphabetically. This means that np.einsum('ij', a) doesn't affect a 2D array, while np.einsum('ji', a) takes its transpose. Additionally, np.einsum('ij,jk', a, b) returns a matrix multiplication, while, np. einsum('ij,jh', a, b) returns the transpose of the multiplication since subscript 'h' precedes subscript 'i'.

In *explicit mode* the output can be directly controlled by specifying output subscript labels. This requires the identifier '->' as well as the list of output subscript labels. This feature increases the flexibility of the function since summing can be disabled or forced when required. The call np.einsum('i->', a) is like *np.sum(a)* if a is a 1-D array, and np.einsum('ii->i', a) is like *np.diag(a)* if a is a square 2-D array. The difference is that *einsum* does not allow broadcasting by default. Additionally np.einsum('ij,jh->ih', a, b) directly specifies the order of the output subscript labels and therefore returns matrix multiplication, unlike the example above in implicit mode.

To enable and control broadcasting, use an ellipsis. Default NumPy-style broadcasting is done by adding an ellipsis to the left of each term, like np.einsum('...ii->...i', a). np.einsum('...i->...', a) is like *np.sum(a, axis=-1)* for array a of any shape. To take the trace along the first and last axes, you can do np.einsum('i...i', a), or to do a matrix-matrix product with the left-most indices instead of rightmost, one can do np.einsum('ij...,jk...->ik...', a, b).

When there is only one operand, no axes are summed, and no output parameter is provided, a view into the operand is returned instead of a new array. Thus, taking the diagonal as np.einsum('ii->i', a) produces a view (changed in version 1.10.0).

*einsum* also provides an alternative way to provide the subscripts and operands as einsum(op0, sublist0, op1, sublist1, ..., [sublistout]). If the output shape is not provided in this format *einsum* will be calculated in implicit mode, otherwise it will be performed explicitly. The examples below have corresponding *einsum* calls with the two parameter methods.

Views returned from einsum are now writeable whenever the input array is writeable. For example, np. einsum('ijk...->kji...', a) will now have the same effect as *np.swapaxes(a, 0, 2)* and np.einsum('ii->i', a) will return a writeable view of the diagonal of a 2D array.

Added the optimize argument which will optimize the contraction order of an einsum expression. For a contraction with three or more operands this can greatly increase the computational efficiency at the cost of a larger memory footprint during computation.

Typically a 'greedy' algorithm is applied which empirical tests have shown returns the optimal path in the majority of cases. In some cases 'optimal' will return the superlative path through a more expensive, exhaustive search. For iterative calculations it may be advisable to calculate the optimal path once and reuse that path by supplying it as an argument. An example is given below.

See *numpy.einsum_path* for more details.

### **Examples**

**>>>** a = np.arange(25).reshape(5,5) **>>>** b = np.arange(5) **>>>** c = np.arange(6).reshape(2,3)

Trace of a matrix:

```
>>> np.einsum('ii', a)
60
>>> np.einsum(a, [0,0])
60
>>> np.trace(a)
60
```
Extract the diagonal (requires explicit form):

```
>>> np.einsum('ii->i', a)
array([ 0, 6, 12, 18, 24])
>>> np.einsum(a, [0,0], [0])
array([ 0, 6, 12, 18, 24])
>>> np.diag(a)
array([ 0, 6, 12, 18, 24])
```
Sum over an axis (requires explicit form):

```
>>> np.einsum('ij->i', a)
array([ 10, 35, 60, 85, 110])
>>> np.einsum(a, [0,1], [0])
array([ 10, 35, 60, 85, 110])
>>> np.sum(a, axis=1)
array([ 10, 35, 60, 85, 110])
```
For higher dimensional arrays summing a single axis can be done with ellipsis:

```
>>> np.einsum('...j->...', a)
array([ 10, 35, 60, 85, 110])
>>> np.einsum(a, [Ellipsis,1], [Ellipsis])
array([ 10, 35, 60, 85, 110])
```
Compute a matrix transpose, or reorder any number of axes:

```
>>> np.einsum('ji', c)
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> np.einsum('ij->ji', c)
array([[0, 3],
       [1, 4],
```
(continues on next page)

(continued from previous page)

```
[2, 5]])
>>> np.einsum(c, [1,0])
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> np.transpose(c)
array([[0, 3],
       [1, 4],
       [2, 5]])
```
Vector inner products:

**>>>** np.einsum('i,i', b, b) 30 **>>>** np.einsum(b, [0], b, [0]) 30 **>>>** np.inner(b,b) 30

Matrix vector multiplication:

```
>>> np.einsum('ij,j', a, b)
array([ 30, 80, 130, 180, 230])
>>> np.einsum(a, [0,1], b, [1])
array([ 30, 80, 130, 180, 230])
>>> np.dot(a, b)
array([ 30, 80, 130, 180, 230])
>>> np.einsum('...j,j', a, b)
array([ 30, 80, 130, 180, 230])
```
Broadcasting and scalar multiplication:

```
>>> np.einsum('..., ...', 3, c)
array([[ 0, 3, 6],
       [ 9, 12, 15]])
>>> np.einsum(',ij', 3, c)
array([[ 0, 3, 6],
       [ 9, 12, 15]])
>>> np.einsum(3, [Ellipsis], c, [Ellipsis])
array([[ 0, 3, 6],
       [ 9, 12, 15]])
>>> np.multiply(3, c)
array([[ 0, 3, 6],
       [ 9, 12, 15]])
```
Vector outer product:

```
>>> np.einsum('i,j', np.arange(2)+1, b)
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])
>>> np.einsum(np.arange(2)+1, [0], b, [1])
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])
>>> np.outer(np.arange(2)+1, b)
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])
```
Tensor contraction:

```
>>> a = np.arange(60.).reshape(3,4,5)
>>> b = np.arange(24.).reshape(4,3,2)
>>> np.einsum('ijk,jil->kl', a, b)
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
>>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
>>> np.tensordot(a,b, axes=([1,0],[0,1]))
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
```
Writeable returned arrays (since version 1.10.0):

```
>>> a = np.zeros((3, 3))
>>> np.einsum('ii->i', a)[:] = 1
>>> a
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```
Example of ellipsis use:

```
>>> a = np.arange(6).reshape((3,2))
>>> b = np.arange(12).reshape((4,3))
>>> np.einsum('ki,jk->ij', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])
>>> np.einsum('ki,...k->i...', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])
>>> np.einsum('k...,jk', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])
```
Chained array operations. For more complicated contractions, speed ups might be achieved by repeatedly computing a 'greedy' path or pre-computing the 'optimal' path and repeatedly applying it, using an *einsum_path* insertion (since version 1.12.0). Performance improvements can be particularly significant with larger arrays:

```
>>> a = np.ones(64).reshape(2,4,8)
```
Basic *einsum*: ~1520ms (benchmarked on 3.1GHz Intel i5.)

```
>>> for iteration in range(500):
... _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)
```
Sub-optimal *einsum* (due to repeated path calculation time): ~330ms

```
>>> for iteration in range(500):
... _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,
... optimize='optimal')
```
Greedy *einsum* (faster optimal path approximation): ~160ms

```
>>> for iteration in range(500):
... _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='greedy')
```
Optimal *einsum* (best usage pattern in some use cases): ~110ms

```
>>> path = np.einsum_path('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,
... optimize='optimal')[0]
>>> for iteration in range(500):
... _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)
```
#### numpy.**einsum_path**(*subscripts*, **operands*, *optimize='greedy'*)

Evaluates the lowest cost contraction order for an einsum expression by considering the creation of intermediate arrays.

#### **Parameters**

#### **subscripts**

[str] Specifies the subscripts for summation.

#### ***operands**

[list of array_like] These are the arrays for the operation.

#### **optimize**

[{bool, list, tuple, 'greedy', 'optimal'}] Choose the type of path. If a tuple is provided, the second argument is assumed to be the maximum intermediate size created. If only a single argument is provided the largest input or output array size is used as a maximum intermediate size.

- if a list is given that starts with einsum_path, uses this as the contraction path
- if False no optimization is taken
- if True defaults to the 'greedy' algorithm
- 'optimal' An algorithm that combinatorially explores all possible ways of contracting the listed tensors and chooses the least costly path. Scales exponentially with the number of terms in the contraction.
- 'greedy' An algorithm that chooses the best pair contraction at each step. Effectively, this algorithm searches the largest inner, Hadamard, and then outer products at each step. Scales cubically with the number of terms in the contraction. Equivalent to the 'optimal' path for most contractions.

Default is 'greedy'.

### **Returns**

#### **path**

[list of tuples] A list representation of the einsum path.

### **string_repr**

[str] A printable representation of the einsum path.

**See also:**

*einsum***,** *linalg.multi_dot*

#### **Notes**

The resulting path indicates which terms of the input contraction should be contracted first, the result of this contraction is then appended to the end of the contraction list. This list can then be iterated over until all intermediate contractions are complete.

### **Examples**

We can begin with a chain dot example. In this case, it is optimal to contract the b and c tensors first as represented by the first element of the path (1, 2). The resulting tensor is added to the end of the contraction and the remaining contraction (0, 1) is then completed.

```
>>> np.random.seed(123)
>>> a = np.random.rand(2, 2)
>>> b = np.random.rand(2, 5)
>>> c = np.random.rand(5, 2)
>>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
>>> print(path_info[0])
['einsum_path', (1, 2), (0, 1)]
>>> print(path_info[1])
 Complete contraction: ij,jk,kl->il # may vary
       Naive scaling: 4
    Optimized scaling: 3
     Naive FLOP count: 1.600e+02
 Optimized FLOP count: 5.600e+01
  Theoretical speedup: 2.857
 Largest intermediate: 4.000e+00 elements
-------------------------------------------------------------------------
scaling current remaining
-------------------------------------------------------------------------
  3 kl,jk->jl ij,jl->il
  3 jl,ij->il il->il
```
A more complex index transformation example.

```
>>> I = np.random.rand(10, 10, 10, 10)
>>> C = np.random.rand(10, 10)
>>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,
... optimize='greedy')
```

```
>>> print(path_info[0])
['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]
>>> print(path_info[1])
 Complete contraction: ea,fb,abcd,gc,hd->efgh # may vary
        Naive scaling: 8
    Optimized scaling: 5
     Naive FLOP count: 8.000e+08
 Optimized FLOP count: 8.000e+05
  Theoretical speedup: 1000.000
 Largest intermediate: 1.000e+04 elements
--------------------------------------------------------------------------
scaling current remaining
--------------------------------------------------------------------------
```
(continues on next page)

(continued from previous page)

| 5 | abcd,ea->bcde | fb,gc,hd,bcde->efgh |
| --- | --- | --- |
| 5 | bcde,fb->cdef | gc,hd,cdef->efgh |
| 5 | cdef,gc->defg | hd,defg->efgh |
| 5 | defg,hd->efgh | efgh->efgh |

linalg.**matrix_power**(*a*, *n*)

Raise a square matrix to the (integer) power *n*.

For positive integers *n*, the power is computed by repeated matrix squarings and matrix multiplications. If n == 0, the identity matrix of the same shape as M is returned. If n < 0, the inverse is computed and then raised to the abs(n).

**Note:** Stacks of object matrices are not currently supported.

#### **Parameters**

**a**

[(…, M, M) array_like] Matrix to be "powered".

**n**

[int] The exponent can be any integer or long integer, positive, negative, or zero.

### **Returns**

#### **a**n**

[(…, M, M) ndarray or matrix object] The return value is the same shape and type as *M*; if the exponent is positive or zero then the type of the elements is the same as those of *M*. If the exponent is negative the elements are floating-point.

### **Raises**

#### **LinAlgError**

For matrices that are not square or that (for negative powers) cannot be inverted numerically.

#### **Examples**

```
>>> import numpy as np
>>> from numpy.linalg import matrix_power
>>> i = np.array([[0, 1], [-1, 0]]) # matrix equiv. of the imaginary unit
>>> matrix_power(i, 3) # should = -i
array([[ 0, -1],
       [ 1, 0]])
>>> matrix_power(i, 0)
array([[1, 0],
       [0, 1]])
>>> matrix_power(i, -3) # should = 1/(-i) = i, but w/ f.p. elements
array([[ 0., 1.],
       [-1., 0.]])
```
Somewhat more sophisticated example

```
>>> q = np.zeros((4, 4))
>>> q[0:2, 0:2] = -i
>>> q[2:4, 2:4] = i
```
(continues on next page)

(continued from previous page)

```
>>> q # one of the three quaternion units not equal to 1
array([[ 0., -1., 0., 0.],
      [ 1., 0., 0., 0.],
      [ 0., 0., 0., 1.],
      [ 0., 0., -1., 0.]])
>>> matrix_power(q, 2) # = -np.eye(4)
array([[-1., 0., 0., 0.],
      [ 0., -1., 0., 0.],
      [ 0., 0., -1., 0.],
      [ 0., 0., 0., -1.]])
```
numpy.**kron**(*a*, *b*)

Kronecker product of two arrays.

Computes the Kronecker product, a composite array made of blocks of the second array scaled by the first.

#### **Parameters**

**a, b** [array_like]

### **Returns**

**out**

[ndarray]

**See also:**

#### *outer*

The outer product

### **Notes**

The function assumes that the number of dimensions of *a* and *b* are the same, if necessary prepending the smallest with ones. If a.shape = (r0,r1,..,rN) and b.shape = (s0,s1,...,sN), the Kronecker product has shape (r0*s0, r1*s1, ..., rN*SN). The elements are products of elements from *a* and *b*, organized explicitly by:

```
kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]
```
where:

kt = it * st + jt, t = 0,...,N

In the common 2-D case (N=1), the block structure can be visualized:

```
[[ a[0,0]*b, a[0,1]*b, ... , a[0,-1]*b ],
[ ... ... ],
[ a[-1,0]*b, a[-1,1]*b, ... , a[-1,-1]*b ]]
```
### **Examples**

```
>>> import numpy as np
>>> np.kron([1,10,100], [5,6,7])
array([ 5, 6, 7, ..., 500, 600, 700])
>>> np.kron([5,6,7], [1,10,100])
array([ 5, 50, 500, ..., 7, 70, 700])
```

```
>>> np.kron(np.eye(2), np.ones((2,2)))
array([[1., 1., 0., 0.],
      [1., 1., 0., 0.],
      [0., 0., 1., 1.],
      [0., 0., 1., 1.]])
```

```
>>> a = np.arange(100).reshape((2,5,2,5))
>>> b = np.arange(24).reshape((2,3,4))
>>> c = np.kron(a,b)
>>> c.shape
(2, 10, 6, 20)
>>> I = (1,3,0,2)
>>> J = (0,2,1)
>>> J1 = (0,) + J # extend to ndim=4
>>> S1 = (1,) + b.shape
>>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))
>>> c[K] == a[I]*b[J]
True
```
linalg.**cross**(*x1*, *x2*, */* , ***, *axis=-1*)

Returns the cross product of 3-element vectors.

If x1 and/or x2 are multi-dimensional arrays, then the cross-product of each pair of corresponding 3-element vectors is independently computed.

This function is Array API compatible, contrary to *numpy.cross*.

#### **Parameters**

### **x1**

[array_like] The first input array.

#### **x2**

[array_like] The second input array. Must be compatible with x1 for all non-compute axes. The size of the axis over which to compute the cross-product must be the same size as the respective axis in x1.

#### **axis**

[int, optional] The axis (dimension) of x1 and x2 containing the vectors for which to compute the cross-product. Default: -1.

#### **Returns**

### **out**

[ndarray] An array containing the cross products.

#### **See also:**

*numpy.cross*

### **Examples**

Vector cross-product.

**>>>** x = np.array([1, 2, 3]) **>>>** y = np.array([4, 5, 6]) **>>>** np.linalg.cross(x, y) array([-3, 6, -3])

Multiple vector cross-products. Note that the direction of the cross product vector is defined by the *right-hand rule*.

```
>>> x = np.array([[1,2,3], [4,5,6]])
>>> y = np.array([[4,5,6], [1,2,3]])
>>> np.linalg.cross(x, y)
array([[-3, 6, -3],
       [ 3, -6, 3]])
```

```
>>> x = np.array([[1, 2], [3, 4], [5, 6]])
>>> y = np.array([[4, 5], [6, 1], [2, 3]])
>>> np.linalg.cross(x, y, axis=0)
array([[-24, 6],
       [ 18, 24],
       [-6, -18]])
```
### **Decompositions**

| linalg.cholesky(a, /, *[, upper]) | Cholesky decomposition. |
| --- | --- |
| linalg.outer(x1, x2, /) | Compute the outer product of two vectors. |
| linalg.qr(a[, mode]) | Compute the qr factorization of a matrix. |
| linalg.svd(a[, full_matrices, compute_uv, ...]) | Singular Value Decomposition. |
| linalg.svdvals(x, /) | Returns the singular values of a matrix (or a stack of ma |
|  | trices) x. |

### linalg.**cholesky**(*a*, */* , ***, *upper=False*)

Cholesky decomposition.

Return the lower or upper Cholesky decomposition, L * L.H or U.H * U, of the square matrix a, where L is lower-triangular, U is upper-triangular, and .H is the conjugate transpose operator (which is the ordinary transpose if a is real-valued). a must be Hermitian (symmetric if real-valued) and positive-definite. No checking is performed to verify whether a is Hermitian or not. In addition, only the lower or upper-triangular and diagonal elements of a are used. Only L or U is actually returned.

### **Parameters**

### **a**

[(…, M, M) array_like] Hermitian (symmetric if all elements are real), positive-definite input matrix.

### **upper**

[bool] If True, the result must be the upper-triangular Cholesky factor. If False, the result must be the lower-triangular Cholesky factor. Default: False.

### **Returns**

**L**

[(…, M, M) array_like] Lower or upper-triangular Cholesky factor of *a*. Returns a matrix object if *a* is a matrix object.

#### **Raises**

**LinAlgError** If the decomposition fails, for example, if *a* is not positive-definite.

### **See also:**

```
scipy.linalg.cholesky
    Similar function in SciPy.
```
**scipy.linalg.cholesky_banded** Cholesky decompose a banded Hermitian positive-definite matrix.

**scipy.linalg.cho_factor** Cholesky decomposition of a matrix, to use in scipy.linalg.cho_solve.

#### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

The Cholesky decomposition is often used as a fast way of solving

*A***x** = **b**

(when *A* is both Hermitian/symmetric and positive-definite).

First, we solve for **y** in

```
Ly = b,
```
and then for **x** in

```
L
 Hx = y.
```
#### **Examples**

```
>>> import numpy as np
>>> A = np.array([[1,-2j],[2j,5]])
>>> A
array([[ 1.+0.j, -0.-2.j],
       [ 0.+2.j, 5.+0.j]])
>>> L = np.linalg.cholesky(A)
>>> L
array([[1.+0.j, 0.+0.j],
       [0.+2.j, 1.+0.j]])
>>> np.dot(L, L.T.conj()) # verify that L * L.H = A
array([[1.+0.j, 0.-2.j],
       [0.+2.j, 5.+0.j]])
>>> A = [[1,-2j],[2j,5]] # what happens if A is only array_like?
>>> np.linalg.cholesky(A) # an ndarray object is returned
array([[1.+0.j, 0.+0.j],
       [0.+2.j, 1.+0.j]])
>>> # But a matrix object is returned if A is a matrix object
>>> np.linalg.cholesky(np.matrix(A))
matrix([[ 1.+0.j, 0.+0.j],
        [ 0.+2.j, 1.+0.j]])
```
(continues on next page)

(continued from previous page)

```
>>> # The upper-triangular Cholesky factor can also be obtained.
>>> np.linalg.cholesky(A, upper=True)
array([[1.-0.j, 0.-2.j],
       [0.-0.j, 1.-0.j]])
```
linalg.**outer**(*x1*, *x2*, */* )

Compute the outer product of two vectors.

This function is Array API compatible. Compared to np.outer it accepts 1-dimensional inputs only.

### **Parameters**

**x1**

[(M,) array_like] One-dimensional input array of size N. Must have a numeric data type.

**x2**

[(N,) array_like] One-dimensional input array of size M. Must have a numeric data type.

#### **Returns**

**out**

[(M, N) ndarray] out[i, j] = a[i] * b[j]

**See also:**

*outer*

### **Examples**

Make a (*very* coarse) grid for computing a Mandelbrot set:

```
>>> rl = np.linalg.outer(np.ones((5,)), np.linspace(-2, 2, 5))
>>> rl
array([[-2., -1., 0., 1., 2.],
      [-2., -1., 0., 1., 2.],
      [-2., -1., 0., 1., 2.],
      [-2., -1., 0., 1., 2.],
      [-2., -1., 0., 1., 2.]])
>>> im = np.linalg.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
>>> im
array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
      [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
      [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
      [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
      [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
>>> grid = rl + im
>>> grid
array([[-2.+2.j, -1.+2.j, 0.+2.j, 1.+2.j, 2.+2.j],
      [-2.+1.j, -1.+1.j, 0.+1.j, 1.+1.j, 2.+1.j],
      [-2.+0.j, -1.+0.j, 0.+0.j, 1.+0.j, 2.+0.j],
      [-2.-1.j, -1.-1.j, 0.-1.j, 1.-1.j, 2.-1.j],
      [-2.-2.j, -1.-2.j, 0.-2.j, 1.-2.j, 2.-2.j]])
```
An example using a "vector" of letters:

```
>>> x = np.array(['a', 'b', 'c'], dtype=object)
>>> np.linalg.outer(x, [1, 2, 3])
```
(continues on next page)

(continued from previous page)

```
array([['a', 'aa', 'aaa'],
       ['b', 'bb', 'bbb'],
       ['c', 'cc', 'ccc']], dtype=object)
```
linalg.**qr**(*a*, *mode='reduced'*)

Compute the qr factorization of a matrix.

Factor the matrix *a* as *qr*, where *q* is orthonormal and *r* is upper-triangular.

#### **Parameters**

**a**

[array_like, shape (…, M, N)] An array-like object with the dimensionality of at least 2.

#### **mode**

[{'reduced', 'complete', 'r', 'raw'}, optional, default: 'reduced'] If K = min(M, N), then

- 'reduced' : returns Q, R with dimensions (…, M, K), (…, K, N)
- 'complete' : returns Q, R with dimensions (…, M, M), (…, M, N)
- 'r' : returns R only with dimensions (…, K, N)
- 'raw' : returns h, tau with dimensions (…, N, M), (…, K,)

The options 'reduced', 'complete, and 'raw' are new in numpy 1.8, see the notes for more information. The default is 'reduced', and to maintain backward compatibility with earlier versions of numpy both it and the old default 'full' can be omitted. Note that array h returned in 'raw' mode is transposed for calling Fortran. The 'economic' mode is deprecated. The modes 'full' and 'economic' may be passed using only the first letter for backwards compatibility, but all others must be spelled out. See the Notes for more explanation.

#### **Returns**

### **When mode is 'reduced' or 'complete', the result will be a namedtuple with the attributes** *Q* **and** *R***.**

### **Q**

[ndarray of float or complex, optional] A matrix with orthonormal columns. When mode = 'complete' the result is an orthogonal/unitary matrix depending on whether or not a is real/complex. The determinant may be either +/- 1 in that case. In case the number of dimensions in the input array is greater than 2 then a stack of the matrices with above properties is returned.

#### **R**

[ndarray of float or complex, optional] The upper-triangular matrix or a stack of uppertriangular matrices if the number of dimensions in the input array is greater than 2.

#### **(h, tau)**

[ndarrays of np.double or np.cdouble, optional] The array h contains the Householder reflectors that generate q along with r. The tau array contains scaling factors for the reflectors. In the deprecated 'economic' mode only h is returned.

### **Raises**

**LinAlgError** If factoring fails.

#### **See also:**

```
scipy.linalg.qr
```
Similar function in SciPy.

```
scipy.linalg.rq
```
Compute RQ decomposition of a matrix.

### **Notes**

This is an interface to the LAPACK routines dgeqrf, zgeqrf, dorgqr, and zungqr.

For more information on the qr factorization, see for example: https://en.wikipedia.org/wiki/QR_factorization

Subclasses of *ndarray* are preserved except for the 'raw' mode. So if *a* is of type *matrix*, all the return values will be matrices too.

New 'reduced', 'complete', and 'raw' options for mode were added in NumPy 1.8.0 and the old option 'full' was made an alias of 'reduced'. In addition the options 'full' and 'economic' were deprecated. Because 'full' was the previous default and 'reduced' is the new default, backward compatibility can be maintained by letting *mode* default. The 'raw' option was added so that LAPACK routines that can multiply arrays by q using the Householder reflectors can be used. Note that in this case the returned arrays are of type np.double or np.cdouble and the h array is transposed to be FORTRAN compatible. No routines using the 'raw' return are currently exposed by numpy, but some are available in lapack_lite and just await the necessary work.

### **Examples**

```
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> a = rng.normal(size=(9, 6))
>>> Q, R = np.linalg.qr(a)
>>> np.allclose(a, np.dot(Q, R)) # a does equal QR
True
>>> R2 = np.linalg.qr(a, mode='r')
>>> np.allclose(R, R2) # mode='r' returns the same R as mode='full'
True
>>> a = np.random.normal(size=(3, 2, 2)) # Stack of 2 x 2 matrices as input
>>> Q, R = np.linalg.qr(a)
>>> Q.shape
(3, 2, 2)
>>> R.shape
(3, 2, 2)
>>> np.allclose(a, np.matmul(Q, R))
True
```
Example illustrating a common use of *qr*: solving of least squares problems

What are the least-squares-best *m* and *y0* in y = y0 + mx for the following data: {(0,1), (1,0), (1,2), (2,1)}. (Graph the points and you'll see that it should be y0 = 0, m = 1.) The answer is provided by solving the overdetermined matrix equation Ax = b, where:

A = array([[0, 1], [1, 1], [1, 1], [2, 1]]) x = array([[y0], [m]]) b = array([[1], [0], [2], [1]])

If A = QR such that Q is orthonormal (which is always possible via Gram-Schmidt), then x = inv(R) * (Q.T) * b. (In numpy practice, however, we simply use *lstsq*.)

```
>>> A = np.array([[0, 1], [1, 1], [1, 1], [2, 1]])
>>> A
array([[0, 1],
```
(continues on next page)

(continued from previous page)

```
[1, 1],
       [1, 1],
       [2, 1]])
>>> b = np.array([1, 2, 2, 3])
>>> Q, R = np.linalg.qr(A)
>>> p = np.dot(Q.T, b)
>>> np.dot(np.linalg.inv(R), p)
array([ 1., 1.])
```
linalg.**svd**(*a*, *full_matrices=True*, *compute_uv=True*, *hermitian=False*)

Singular Value Decomposition.

When *a* is a 2D array, and full_matrices=False, then it is factorized as u @ np.diag(s) @ vh = (u * s) @ vh, where *u* and the Hermitian transpose of *vh* are 2D arrays with orthonormal columns and *s* is a 1D array of *a*'s singular values. When *a* is higher-dimensional, SVD is applied in stacked mode as explained below.

#### **Parameters**

**a**

[(…, M, N) array_like] A real or complex array with a.ndim >= 2.

#### **full_matrices**

[bool, optional] If True (default), *u* and *vh* have the shapes (..., M, M) and (..., N, N), respectively. Otherwise, the shapes are (..., M, K) and (..., K, N), respectively, where K = min(M, N).

### **compute_uv**

[bool, optional] Whether or not to compute *u* and *vh* in addition to *s*. True by default.

#### **hermitian**

[bool, optional] If True, *a* is assumed to be Hermitian (symmetric if real-valued), enabling a more efficient method for finding singular values. Defaults to False.

#### **Returns**

### **When** *compute_uv* **is True, the result is a namedtuple with the following attribute names:**

#### **U**

[{ (…, M, M), (…, M, K) } array] Unitary array(s). The first a.ndim - 2 dimensions have the same size as those of the input *a*. The size of the last two dimensions depends on the value of *full_matrices*. Only returned when *compute_uv* is True.

#### **S**

[(…, K) array] Vector(s) with the singular values, within each vector sorted in descending order. The first a.ndim - 2 dimensions have the same size as those of the input *a*.

#### **Vh**

[{ (…, N, N), (…, K, N) } array] Unitary array(s). The first a.ndim - 2 dimensions have the same size as those of the input *a*. The size of the last two dimensions depends on the value of *full_matrices*. Only returned when *compute_uv* is True.

#### **Raises**

#### **LinAlgError**

If SVD computation does not converge.

#### **See also:**

```
scipy.linalg.svd
```
Similar function in SciPy.

**scipy.linalg.svdvals**

Compute singular values of a matrix.

### **Notes**

The decomposition is performed using LAPACK routine _gesdd.

SVD is usually described for the factorization of a 2D matrix *A*. The higher-dimensional case will be discussed below. In the 2D case, SVD is written as *A* = *USV H*, where *A* = *a*, *U* = *u*, *S* = np*.*diag(*s*) and *V H* = *vh*. The 1D array *s* contains the singular values of *a* and *u* and *vh* are unitary. The rows of *vh* are the eigenvectors of *AHA* and the columns of *u* are the eigenvectors of *AAH*. In both cases the corresponding (possibly non-zero) eigenvalues are given by s**2.

If *a* has more than two dimensions, then broadcasting rules apply, as explained in *Linear algebra on several matrices at once*. This means that SVD is working in "stacked" mode: it iterates over all indices of the first a.ndim - 2 dimensions and for each combination SVD is applied to the last two indices. The matrix *a* can be reconstructed from the decomposition with either (u * s[..., None, :]) @ vh or u @ (s[..., None] * vh). (The @ operator can be replaced by the function np.matmul for python versions below 3.5.)

If *a* is a matrix object (as opposed to an ndarray), then so are all the return values.

### **Examples**

```
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> a = rng.normal(size=(9, 6)) + 1j*rng.normal(size=(9, 6))
>>> b = rng.normal(size=(2, 7, 8, 3)) + 1j*rng.normal(size=(2, 7, 8, 3))
```
Reconstruction based on full SVD, 2D case:

```
>>> U, S, Vh = np.linalg.svd(a, full_matrices=True)
>>> U.shape, S.shape, Vh.shape
((9, 9), (6,), (6, 6))
>>> np.allclose(a, np.dot(U[:, :6] * S, Vh))
True
>>> smat = np.zeros((9, 6), dtype=complex)
>>> smat[:6, :6] = np.diag(S)
>>> np.allclose(a, np.dot(U, np.dot(smat, Vh)))
True
```
Reconstruction based on reduced SVD, 2D case:

```
>>> U, S, Vh = np.linalg.svd(a, full_matrices=False)
>>> U.shape, S.shape, Vh.shape
((9, 6), (6,), (6, 6))
>>> np.allclose(a, np.dot(U * S, Vh))
True
>>> smat = np.diag(S)
>>> np.allclose(a, np.dot(U, np.dot(smat, Vh)))
True
```
Reconstruction based on full SVD, 4D case:

```
>>> U, S, Vh = np.linalg.svd(b, full_matrices=True)
>>> U.shape, S.shape, Vh.shape
```
(continues on next page)

(continued from previous page)

```
((2, 7, 8, 8), (2, 7, 3), (2, 7, 3, 3))
>>> np.allclose(b, np.matmul(U[..., :3] * S[..., None, :], Vh))
True
>>> np.allclose(b, np.matmul(U[..., :3], S[..., None] * Vh))
True
```
#### Reconstruction based on reduced SVD, 4D case:

```
>>> U, S, Vh = np.linalg.svd(b, full_matrices=False)
>>> U.shape, S.shape, Vh.shape
((2, 7, 8, 3), (2, 7, 3), (2, 7, 3, 3))
>>> np.allclose(b, np.matmul(U * S[..., None, :], Vh))
True
>>> np.allclose(b, np.matmul(U, S[..., None] * Vh))
True
```
### linalg.**svdvals**(*x*, */* )

Returns the singular values of a matrix (or a stack of matrices) x. When x is a stack of matrices, the function will compute the singular values for each matrix in the stack.

This function is Array API compatible.

Calling np.svdvals(x) to get singular values is the same as np.svd(x, compute_uv=False, hermitian=False).

#### **Parameters**

**x**

[(…, M, N) array_like] Input array having shape (…, M, N) and whose last two dimensions form matrices on which to perform singular value decomposition. Should have a floating-point data type.

#### **Returns**

# **out**

[ndarray] An array with shape (…, K) that contains the vector(s) of singular values of length K, where K = min(M, N).

#### **See also:**

```
scipy.linalg.svdvals
```
Compute singular values of a matrix.

#### **Examples**

```
>>> np.linalg.svdvals([[1, 2, 3, 4, 5],
... [1, 4, 9, 16, 25],
... [1, 8, 27, 64, 125]])
array([146.68862757, 5.57510612, 0.60393245])
```
Determine the rank of a matrix using singular values:

```
>>> s = np.linalg.svdvals([[1, 2, 3],
... [2, 4, 6],
... [-1, 1, -1]]); s
array([8.38434191e+00, 1.64402274e+00, 2.31534378e-16])
```
(continues on next page)

(continued from previous page)

```
>>> np.count_nonzero(s > 1e-10) # Matrix of rank 2
2
```
### **Matrix eigenvalues**

| linalg.eig(a) | Compute the eigenvalues and right eigenvectors of a |
| --- | --- |
|  | square array. |
| linalg.eigh(a[, UPLO]) | Return the eigenvalues and eigenvectors of a complex |
|  | Hermitian (conjugate symmetric) or a real symmetric ma |
|  | trix. |
| linalg.eigvals(a) | Compute the eigenvalues of a general matrix. |
| linalg.eigvalsh(a[, UPLO]) | Compute the eigenvalues of a complex Hermitian or real |
|  | symmetric matrix. |

### linalg.**eig**(*a*)

Compute the eigenvalues and right eigenvectors of a square array.

### **Parameters**

**a**

[(…, M, M) array] Matrices for which the eigenvalues and right eigenvectors will be computed

#### **Returns**

### **A namedtuple with the following attributes:**

### **eigenvalues**

[(…, M) array] The eigenvalues, each repeated according to its multiplicity. The eigenvalues are not necessarily ordered. The resulting array will be of complex type, unless the imaginary part is zero in which case it will be cast to a real type. When *a* is real the resulting eigenvalues will be real (0 imaginary part) or occur in conjugate pairs

#### **eigenvectors**

[(…, M, M) array] The normalized (unit "length") eigenvectors, such that the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

### **Raises**

#### **LinAlgError**

If the eigenvalue computation does not converge.

#### **See also:**

### *eigvals*

eigenvalues of a non-symmetric array.

#### *eigh*

eigenvalues and eigenvectors of a real symmetric or complex Hermitian (conjugate symmetric) array.

### *eigvalsh*

eigenvalues of a real symmetric or complex Hermitian (conjugate symmetric) array.

```
scipy.linalg.eig
```
Similar function in SciPy that also solves the generalized eigenvalue problem.

```
scipy.linalg.schur
```
Best choice for unitary and other non-Hermitian normal matrices.

### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

This is implemented using the _geev LAPACK routines which compute the eigenvalues and eigenvectors of general square arrays.

The number *w* is an eigenvalue of *a* if there exists a vector *v* such that a @ v = w * v. Thus, the arrays *a*, *eigenvalues*, and *eigenvectors* satisfy the equations a @ eigenvectors[:,i] = eigenvalues[i] * eigenvectors[:,i] for *i ∈ {*0*, ..., M −* 1*}*.

The array *eigenvectors* may not be of maximum rank, that is, some of the columns may be linearly dependent, although round-off error may obscure that fact. If the eigenvalues are all different, then theoretically the eigenvectors are linearly independent and *a* can be diagonalized by a similarity transformation using *eigenvectors*, i.e, inv(eigenvectors) @ a @ eigenvectors is diagonal.

For non-Hermitian normal matrices the SciPy function scipy.linalg.schur is preferred because the matrix *eigenvectors* is guaranteed to be unitary, which is not the case when using *eig*. The Schur factorization produces an upper triangular matrix rather than a diagonal matrix, but for normal matrices only the diagonal of the upper triangular matrix is needed, the rest is roundoff error.

Finally, it is emphasized that *eigenvectors* consists of the *right* (as in right-hand side) eigenvectors of *a*. A vector *y* satisfying y.T @ a = z * y.T for some number *z* is called a *left* eigenvector of *a*, and, in general, the left and right eigenvectors of a matrix are not necessarily the (perhaps conjugate) transposes of each other.

#### **References**

G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando, FL, Academic Press, Inc., 1980, Various pp.

#### **Examples**

```
>>> import numpy as np
>>> from numpy import linalg as LA
```
(Almost) trivial example with real eigenvalues and eigenvectors.

```
>>> eigenvalues, eigenvectors = LA.eig(np.diag((1, 2, 3)))
>>> eigenvalues
array([1., 2., 3.])
>>> eigenvectors
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```
Real matrix possessing complex eigenvalues and eigenvectors; note that the eigenvalues are complex conjugates of each other.

```
>>> eigenvalues, eigenvectors = LA.eig(np.array([[1, -1], [1, 1]]))
>>> eigenvalues
array([1.+1.j, 1.-1.j])
>>> eigenvectors
array([[0.70710678+0.j , 0.70710678-0.j ],
      [0. -0.70710678j, 0. +0.70710678j]])
```
Complex-valued matrix with real eigenvalues (but complex-valued eigenvectors); note that a.conj().T == a, i.e., *a* is Hermitian.

```
>>> a = np.array([[1, 1j], [-1j, 1]])
>>> eigenvalues, eigenvectors = LA.eig(a)
>>> eigenvalues
array([2.+0.j, 0.+0.j])
>>> eigenvectors
array([[ 0. +0.70710678j, 0.70710678+0.j ], # may vary
      [ 0.70710678+0.j , -0. +0.70710678j]])
```
Be careful about round-off error!

```
>>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])
>>> # Theor. eigenvalues are 1 +/- 1e-9
>>> eigenvalues, eigenvectors = LA.eig(a)
>>> eigenvalues
array([1., 1.])
>>> eigenvectors
array([[1., 0.],
       [0., 1.]])
```
### linalg.**eigh**(*a*, *UPLO='L'*)

Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.

Returns two objects, a 1-D array containing the eigenvalues of *a*, and a 2-D square array or matrix (depending on the input type) of the corresponding eigenvectors (in columns).

### **Parameters**

### **a**

[(…, M, M) array] Hermitian or real symmetric matrices whose eigenvalues and eigenvectors are to be computed.

### **UPLO**

[{'L', 'U'}, optional] Specifies whether the calculation is done with the lower triangular part of *a* ('L', default) or the upper triangular part ('U'). Irrespective of this value only the real parts of the diagonal will be considered in the computation to preserve the notion of a Hermitian matrix. It therefore follows that the imaginary part of the diagonal will always be treated as zero.

### **Returns**

### **A namedtuple with the following attributes:**

#### **eigenvalues**

[(…, M) ndarray] The eigenvalues in ascending order, each repeated according to its multiplicity.

### **eigenvectors**

[{(…, M, M) ndarray, (…, M, M) matrix}] The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue eigenvalues[i]. Will return a matrix object if *a* is a matrix object.

### **Raises**

### **LinAlgError**

If the eigenvalue computation does not converge.

### **See also:**

#### *eigvalsh*

eigenvalues of real symmetric or complex Hermitian (conjugate symmetric) arrays.

```
eig
```
eigenvalues and right eigenvectors for non-symmetric arrays.

```
eigvals
```
eigenvalues of non-symmetric arrays.

```
scipy.linalg.eigh
```
Similar function in SciPy (but also solves the generalized eigenvalue problem).

### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

The eigenvalues/eigenvectors are computed using LAPACK routines _syevd, _heevd.

The eigenvalues of real symmetric or complex Hermitian matrices are always real. [1] The array *eigenvalues* of (column) eigenvectors is unitary and *a*, *eigenvalues*, and *eigenvectors* satisfy the equations dot(a, eigenvectors[:, i]) = eigenvalues[i] * eigenvectors[:, i].

### **References**

[1]

#### **Examples**

```
>>> import numpy as np
>>> from numpy import linalg as LA
>>> a = np.array([[1, -2j], [2j, 5]])
>>> a
array([[ 1.+0.j, -0.-2.j],
      [ 0.+2.j, 5.+0.j]])
>>> eigenvalues, eigenvectors = LA.eigh(a)
>>> eigenvalues
array([0.17157288, 5.82842712])
>>> eigenvectors
array([[-0.92387953+0.j , -0.38268343+0.j ], # may vary
      [ 0. +0.38268343j, 0. -0.92387953j]])
```

```
>>> (np.dot(a, eigenvectors[:, 0]) -
... eigenvalues[0] * eigenvectors[:, 0]) # verify 1st eigenval/vec pair
array([5.55111512e-17+0.0000000e+00j, 0.00000000e+00+1.2490009e-16j])
>>> (np.dot(a, eigenvectors[:, 1]) -
... eigenvalues[1] * eigenvectors[:, 1]) # verify 2nd eigenval/vec pair
array([0.+0.j, 0.+0.j])
```

```
>>> A = np.matrix(a) # what happens if input is a matrix object
>>> A
matrix([[ 1.+0.j, -0.-2.j],
       [ 0.+2.j, 5.+0.j]])
>>> eigenvalues, eigenvectors = LA.eigh(A)
>>> eigenvalues
array([0.17157288, 5.82842712])
>>> eigenvectors
matrix([[-0.92387953+0.j , -0.38268343+0.j ], # may vary
       [ 0. +0.38268343j, 0. -0.92387953j]])
```

```
>>> # demonstrate the treatment of the imaginary part of the diagonal
>>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])
>>> a
array([[5.+2.j, 9.-2.j],
      [0.+2.j, 2.-1.j]])
>>> # with UPLO='L' this is numerically equivalent to using LA.eig() with:
>>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
>>> b
array([[5.+0.j, 0.-2.j],
      [0.+2.j, 2.+0.j]])
>>> wa, va = LA.eigh(a)
>>> wb, vb = LA.eig(b)
>>> wa
array([1., 6.])
>>> wb
array([6.+0.j, 1.+0.j])
>>> va
array([[-0.4472136 +0.j , -0.89442719+0.j ], # may vary
      [ 0. +0.89442719j, 0. -0.4472136j ]])
>>> vb
array([[ 0.89442719+0.j , -0. +0.4472136j],
      [-0. +0.4472136j, 0.89442719+0.j ]])
```
### linalg.**eigvals**(*a*)

Compute the eigenvalues of a general matrix.

Main difference between *eigvals* and *eig*: the eigenvectors aren't returned.

### **Parameters**

#### **a**

[(…, M, M) array_like] A complex- or real-valued matrix whose eigenvalues will be computed.

### **Returns**

**w**

[(…, M,) ndarray] The eigenvalues, each repeated according to its multiplicity. They are not necessarily ordered, nor are they necessarily real for real matrices.

### **Raises**

### **LinAlgError**

If the eigenvalue computation does not converge.

### **See also:**

### *eig*

eigenvalues and right eigenvectors of general arrays

#### *eigvalsh*

eigenvalues of real symmetric or complex Hermitian (conjugate symmetric) arrays.

#### *eigh*

eigenvalues and eigenvectors of real symmetric or complex Hermitian (conjugate symmetric) arrays.

### **scipy.linalg.eigvals**

Similar function in SciPy.

#### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

This is implemented using the _geev LAPACK routines which compute the eigenvalues and eigenvectors of general square arrays.

### **Examples**

Illustration, using the fact that the eigenvalues of a diagonal matrix are its diagonal elements, that multiplying a matrix on the left by an orthogonal matrix, *Q*, and on the right by *Q.T* (the transpose of *Q*), preserves the eigenvalues of the "middle" matrix. In other words, if *Q* is orthogonal, then Q * A * Q.T has the same eigenvalues as A:

```
>>> import numpy as np
>>> from numpy import linalg as LA
>>> x = np.random.random()
>>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
>>> LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])
(1.0, 1.0, 0.0)
```
Now multiply a diagonal matrix by Q on one side and by Q.T on the other:

```
>>> D = np.diag((-1,1))
>>> LA.eigvals(D)
array([-1., 1.])
>>> A = np.dot(Q, D)
>>> A = np.dot(A, Q.T)
>>> LA.eigvals(A)
array([ 1., -1.]) # random
```
#### linalg.**eigvalsh**(*a*, *UPLO='L'*)

Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

Main difference from eigh: the eigenvectors are not computed.

#### **Parameters**

### **a**

[(…, M, M) array_like] A complex- or real-valued matrix whose eigenvalues are to be computed.

#### **UPLO**

[{'L', 'U'}, optional] Specifies whether the calculation is done with the lower triangular part of *a* ('L', default) or the upper triangular part ('U'). Irrespective of this value only the real parts of the diagonal will be considered in the computation to preserve the notion of a Hermitian matrix. It therefore follows that the imaginary part of the diagonal will always be treated as zero.

### **Returns**

**w**

[(…, M,) ndarray] The eigenvalues in ascending order, each repeated according to its multiplicity.

### **Raises**

### **LinAlgError**

If the eigenvalue computation does not converge.

#### **See also:**

#### *eigh*

eigenvalues and eigenvectors of real symmetric or complex Hermitian (conjugate symmetric) arrays.

#### *eigvals*

eigenvalues of general real or complex arrays.

#### *eig*

eigenvalues and right eigenvectors of general real or complex arrays.

**scipy.linalg.eigvalsh** Similar function in SciPy.

### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

The eigenvalues are computed using LAPACK routines _syevd, _heevd.

#### **Examples**

```
>>> import numpy as np
>>> from numpy import linalg as LA
>>> a = np.array([[1, -2j], [2j, 5]])
>>> LA.eigvalsh(a)
array([ 0.17157288, 5.82842712]) # may vary
```

```
>>> # demonstrate the treatment of the imaginary part of the diagonal
>>> a = np.array([[5+2j, 9-2j], [0+2j, 2-1j]])
>>> a
array([[5.+2.j, 9.-2.j],
       [0.+2.j, 2.-1.j]])
>>> # with UPLO='L' this is numerically equivalent to using LA.eigvals()
>>> # with:
>>> b = np.array([[5.+0.j, 0.-2.j], [0.+2.j, 2.-0.j]])
>>> b
array([[5.+0.j, 0.-2.j],
       [0.+2.j, 2.+0.j]])
>>> wa = LA.eigvalsh(a)
>>> wb = LA.eigvals(b)
>>> wa; wb
array([1., 6.])
array([6.+0.j, 1.+0.j])
```
### **Norms and other numbers**

| linalg.norm(x[, ord, axis, keepdims]) | Matrix or vector norm. |
| --- | --- |
| linalg.matrix_norm(x, /, *[, keepdims, ord]) | Computes the matrix norm of a matrix (or a stack of ma |
|  | trices) x. |
| linalg.vector_norm(x, /, *[, axis, ...]) | Computes the vector norm of a vector (or batch of vec |
|  | tors) x. |
| linalg.cond(x[, p]) | Compute the condition number of a matrix. |
| linalg.det(a) | Compute the determinant of an array. |
| linalg.matrix_rank(A[, tol, hermitian, rtol]) | Return matrix rank of array using SVD method |
| linalg.slogdet(a) | Compute the sign and (natural) logarithm of the determi |
|  | nant of an array. |
| trace(a[, offset, axis1, axis2, dtype, out]) | Return the sum along diagonals of the array. |
| linalg.trace(x, /, *[, offset, dtype]) | Returns the sum along the specified diagonals of a matrix |
|  | (or a stack of matrices) x. |

linalg.**norm**(*x*, *ord=None*, *axis=None*, *keepdims=False*)

Matrix or vector norm.

This function is able to return one of eight different matrix norms, or one of an infinite number of vector norms (described below), depending on the value of the ord parameter.

### **Parameters**

**x**

[array_like] Input array. If *axis* is None, *x* must be 1-D or 2-D, unless *ord* is None. If both *axis* and *ord* are None, the 2-norm of x.ravel will be returned.

#### **ord**

[{int, float, inf, -inf, 'fro', 'nuc'}, optional] Order of the norm (see table under Notes for what values are supported for matrices and vectors respectively). inf means numpy's *inf* object. The default is None.

#### **axis**

[{None, int, 2-tuple of ints}, optional.] If *axis* is an integer, it specifies the axis of *x* along which to compute the vector norms. If *axis* is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed. If *axis* is None then either a vector norm (when *x* is 1-D) or a matrix norm (when *x* is 2-D) is returned. The default is None.

### **keepdims**

[bool, optional] If this is set to True, the axes which are normed over are left in the result as dimensions with size one. With this option the result will broadcast correctly against the original *x*.

### **Returns**

**n**

[float or ndarray] Norm of the matrix or vector(s).

### **See also:**

```
scipy.linalg.norm
```
Similar function in SciPy.

### **Notes**

For values of ord < 1, the result is, strictly speaking, not a mathematical 'norm', but it may still be useful for various numerical purposes.

The following norms can be calculated:

| ord | norm for matrices | norm for vectors |
| --- | --- | --- |
| None | Frobenius norm | 2-norm |
| 'fro' | Frobenius norm | – |
| 'nuc' | nuclear norm | – |
| inf | max(sum(abs(x), axis=1)) | max(abs(x)) |
| -inf | min(sum(abs(x), axis=1)) | min(abs(x)) |
| 0 | – | sum(x != 0) |
| 1 | max(sum(abs(x), axis=0)) | as below |
| -1 | min(sum(abs(x), axis=0)) | as below |
| 2 | 2-norm (largest sing. value) | as below |
| -2 | smallest singular value | as below |
| other | – | sum(abs(x)**ord)**(1./ord) |

The Frobenius norm is given by [1]:

*||A||F* = [P *i,j abs*(*ai,j* ) 2 ] 1/2

The nuclear norm is the sum of the singular values.

Both the Frobenius and nuclear norm orders are only defined for matrices and raise a ValueError when x.ndim != 2.

### **References**

[1]

### **Examples**

```
>>> import numpy as np
>>> from numpy import linalg as LA
>>> a = np.arange(9) - 4
>>> a
array([-4, -3, -2, ..., 2, 3, 4])
>>> b = a.reshape((3, 3))
>>> b
array([[-4, -3, -2],
       [-1, 0, 1],
       [ 2, 3, 4]])
```

```
>>> LA.norm(a)
7.745966692414834
>>> LA.norm(b)
7.745966692414834
>>> LA.norm(b, 'fro')
7.745966692414834
>>> LA.norm(a, np.inf)
```
(continues on next page)

(continued from previous page)

```
4.0
>>> LA.norm(b, np.inf)
9.0
>>> LA.norm(a, -np.inf)
0.0
>>> LA.norm(b, -np.inf)
2.0
```

```
>>> LA.norm(a, 1)
20.0
>>> LA.norm(b, 1)
7.0
>>> LA.norm(a, -1)
-4.6566128774142013e-010
>>> LA.norm(b, -1)
6.0
>>> LA.norm(a, 2)
7.745966692414834
>>> LA.norm(b, 2)
7.3484692283495345
```

```
>>> LA.norm(a, -2)
0.0
>>> LA.norm(b, -2)
1.8570331885190563e-016 # may vary
>>> LA.norm(a, 3)
5.8480354764257312 # may vary
>>> LA.norm(a, -3)
0.0
```
Using the *axis* argument to compute vector norms:

```
>>> c = np.array([[ 1, 2, 3],
... [-1, 1, 4]])
>>> LA.norm(c, axis=0)
array([ 1.41421356, 2.23606798, 5. ])
>>> LA.norm(c, axis=1)
array([ 3.74165739, 4.24264069])
>>> LA.norm(c, ord=1, axis=1)
array([ 6., 6.])
```
Using the *axis* argument to compute matrix norms:

```
>>> m = np.arange(8).reshape(2,2,2)
>>> LA.norm(m, axis=(1,2))
array([ 3.74165739, 11.22497216])
>>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
(3.7416573867739413, 11.224972160321824)
```
linalg.**matrix_norm**(*x*, */* , ***, *keepdims=False*, *ord='fro'*)

Computes the matrix norm of a matrix (or a stack of matrices) x.

This function is Array API compatible.

**Parameters**

**x**

[array_like] Input array having shape (…, M, N) and whose two innermost dimensions form MxN matrices.

#### **keepdims**

[bool, optional] If this is set to True, the axes which are normed over are left in the result as dimensions with size one. Default: False.

#### **ord**

[{1, -1, 2, -2, inf, -inf, 'fro', 'nuc'}, optional] The order of the norm. For details see the table under Notes in *numpy.linalg.norm*.

### **See also:**

#### *numpy.linalg.norm*

Generic norm function

### **Examples**

```
>>> from numpy import linalg as LA
>>> a = np.arange(9) - 4
>>> a
array([-4, -3, -2, ..., 2, 3, 4])
>>> b = a.reshape((3, 3))
>>> b
array([[-4, -3, -2],
       [-1, 0, 1],
       [ 2, 3, 4]])
```

```
>>> LA.matrix_norm(b)
7.745966692414834
>>> LA.matrix_norm(b, ord='fro')
7.745966692414834
>>> LA.matrix_norm(b, ord=np.inf)
9.0
>>> LA.matrix_norm(b, ord=-np.inf)
2.0
```

```
>>> LA.matrix_norm(b, ord=1)
7.0
>>> LA.matrix_norm(b, ord=-1)
6.0
>>> LA.matrix_norm(b, ord=2)
7.3484692283495345
>>> LA.matrix_norm(b, ord=-2)
1.8570331885190563e-016 # may vary
```
linalg.**vector_norm**(*x*, */* , ***, *axis=None*, *keepdims=False*, *ord=2*)

Computes the vector norm of a vector (or batch of vectors) x.

This function is Array API compatible.

#### **Parameters**

**x**

[array_like] Input array.

**axis**

[{None, int, 2-tuple of ints}, optional] If an integer, axis specifies the axis (dimension) along

which to compute vector norms. If an n-tuple, axis specifies the axes (dimensions) along which to compute batched vector norms. If None, the vector norm must be computed over all array values (i.e., equivalent to computing the vector norm of a flattened array). Default: None.

#### **keepdims**

[bool, optional] If this is set to True, the axes which are normed over are left in the result as dimensions with size one. Default: False.

**ord**

[{int, float, inf, -inf}, optional] The order of the norm. For details see the table under Notes in *numpy.linalg.norm*.

#### **See also:**

```
numpy.linalg.norm
```
Generic norm function

#### **Examples**

```
>>> from numpy import linalg as LA
>>> a = np.arange(9) + 1
>>> a
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b = a.reshape((3, 3))
>>> b
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

```
>>> LA.vector_norm(b)
16.881943016134134
>>> LA.vector_norm(b, ord=np.inf)
9.0
>>> LA.vector_norm(b, ord=-np.inf)
1.0
```

```
>>> LA.vector_norm(b, ord=0)
9.0
>>> LA.vector_norm(b, ord=1)
45.0
>>> LA.vector_norm(b, ord=-1)
0.3534857623790153
>>> LA.vector_norm(b, ord=2)
16.881943016134134
>>> LA.vector_norm(b, ord=-2)
0.8058837395885292
```
linalg.**cond**(*x*, *p=None*)

Compute the condition number of a matrix.

This function is capable of returning the condition number using one of seven different norms, depending on the value of *p* (see Parameters below).

**Parameters**

[(…, M, N) array_like] The matrix whose condition number is sought.

#### **p**

**x**

[{None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional] Order of the norm used in the condition number computation:

| p | norm for matrices |
| --- | --- |
| None | 2-norm, computed directly using the SVD |
| 'fro' | Frobenius norm |
| inf | max(sum(abs(x), axis=1)) |
| -inf | min(sum(abs(x), axis=1)) |
| 1 | max(sum(abs(x), axis=0)) |
| -1 | min(sum(abs(x), axis=0)) |
| 2 | 2-norm (largest sing. value) |
| -2 | smallest singular value |

inf means the *numpy.inf* object, and the Frobenius norm is the root-of-sum-of-squares norm.

#### **Returns**

**c**

[{float, inf}] The condition number of the matrix. May be infinite.

#### **See also:**

*numpy.linalg.norm*

#### **Notes**

The condition number of *x* is defined as the norm of *x* times the norm of the inverse of *x* [1]; the norm can be the usual L2-norm (root-of-sum-of-squares) or one of a number of other matrix norms.

#### **References**

[1]

#### **Examples**

```
>>> import numpy as np
>>> from numpy import linalg as LA
>>> a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
>>> a
array([[ 1, 0, -1],
       [ 0, 1, 0],
       [ 1, 0, 1]])
>>> LA.cond(a)
1.4142135623730951
>>> LA.cond(a, 'fro')
3.1622776601683795
>>> LA.cond(a, np.inf)
```
(continues on next page)

(continued from previous page)

```
2.0
>>> LA.cond(a, -np.inf)
1.0
>>> LA.cond(a, 1)
2.0
>>> LA.cond(a, -1)
1.0
>>> LA.cond(a, 2)
1.4142135623730951
>>> LA.cond(a, -2)
0.70710678118654746 # may vary
>>> (min(LA.svd(a, compute_uv=False)) *
... min(LA.svd(LA.inv(a), compute_uv=False)))
0.70710678118654746 # may vary
```

```
linalg.det(a)
```
Compute the determinant of an array.

#### **Parameters**

**a**

[(…, M, M) array_like] Input array to compute determinants for.

#### **Returns**

**det**

[(…) array_like] Determinant of *a*.

### **See also:**

#### *slogdet*

Another way to represent the determinant, more suitable for large matrices where underflow/overflow may occur.

```
scipy.linalg.det
```
Similar function in SciPy.

#### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

The determinant is computed via LU factorization using the LAPACK routine z/dgetrf.

#### **Examples**

The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

```
>>> import numpy as np
>>> a = np.array([[1, 2], [3, 4]])
>>> np.linalg.det(a)
-2.0 # may vary
```
Computing determinants for a stack of matrices:

```
>>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
>>> a.shape
(3, 2, 2)
>>> np.linalg.det(a)
array([-2., -3., -8.])
```
linalg.**matrix_rank**(*A*, *tol=None*, *hermitian=False*, ***, *rtol=None*)

Return matrix rank of array using SVD method

Rank of the array is the number of singular values of the array that are greater than *tol*.

### **Parameters**

### **A**

[{(M,), (…, M, N)} array_like] Input vector or stack of matrices.

### **tol**

[(…) array_like, float, optional] Threshold below which SVD values are considered zero. If *tol* is None, and S is an array with singular values for *M*, and eps is the epsilon value for datatype of S, then *tol* is set to S.max() * max(M, N) * eps.

### **hermitian**

[bool, optional] If True, *A* is assumed to be Hermitian (symmetric if real-valued), enabling a more efficient method for finding singular values. Defaults to False.

### **rtol**

[(…) array_like, float, optional] Parameter for the relative tolerance component. Only tol or rtol can be set at a time. Defaults to max(M, N) * eps.

New in version 2.0.0.

### **Returns**

### **rank**

[(…) array_like] Rank of A.

### **Notes**

The default threshold to detect rank deficiency is a test on the magnitude of the singular values of *A*. By default, we identify singular values less than S.max() * max(M, N) * eps as indicating rank deficiency (with the symbols defined above). This is the algorithm MATLAB uses [1]. It also appears in *Numerical recipes* in the discussion of SVD solutions for linear least squares [2].

This default threshold is designed to detect rank deficiency accounting for the numerical errors of the SVD computation. Imagine that there is a column in *A* that is an exact (in floating point) linear combination of other columns in *A*. Computing the SVD on *A* will not produce a singular value exactly equal to 0 in general: any difference of the smallest SVD value from 0 will be caused by numerical imprecision in the calculation of the SVD. Our threshold for small SVD values takes this numerical imprecision into account, and the default threshold will detect such numerical rank deficiency. The threshold may declare a matrix *A* rank deficient even if the linear combination of some columns of *A* is not exactly equal to another column of *A* but only numerically very close to another column of *A*.

We chose our default threshold because it is in wide use. Other thresholds are possible. For example, elsewhere in the 2007 edition of *Numerical recipes* there is an alternative threshold of S.max() * np.finfo(A.dtype). eps / 2. * np.sqrt(m + n + 1.). The authors describe this threshold as being based on "expected roundoff error" (p 71).

The thresholds above deal with floating point roundoff error in the calculation of the SVD. However, you may have more information about the sources of error in *A* that would make you consider other tolerance values to detect *effective* rank deficiency. The most useful measure of the tolerance depends on the operations you intend to use on your matrix. For example, if your data come from uncertain measurements with uncertainties greater than floating point epsilon, choosing a tolerance near that uncertainty may be preferable. The tolerance may be absolute if the uncertainties are absolute rather than relative.

### **References**

[1], [2]

### **Examples**

```
>>> import numpy as np
>>> from numpy.linalg import matrix_rank
>>> matrix_rank(np.eye(4)) # Full rank matrix
4
>>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
>>> matrix_rank(I)
3
>>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
1
>>> matrix_rank(np.zeros((4,)))
0
```
#### linalg.**slogdet**(*a*)

Compute the sign and (natural) logarithm of the determinant of an array.

If an array has a very small or very large determinant, then a call to *det* may overflow or underflow. This routine is more robust against such issues, because it computes the logarithm of the determinant rather than the determinant itself.

#### **Parameters**

**a**

[(…, M, M) array_like] Input array, has to be a square 2-D array.

#### **Returns**

### **A namedtuple with the following attributes:**

**sign**

[(…) array_like] A number representing the sign of the determinant. For a real matrix, this is 1, 0, or -1. For a complex matrix, this is a complex number with absolute value 1 (i.e., it is on the unit circle), or else 0.

#### **logabsdet**

[(…) array_like] The natural log of the absolute value of the determinant.

**If the determinant is zero, then** *sign* **will be 0 and** *logabsdet* **will be -inf. In all cases, the determinant is equal to sign * np.exp(logabsdet).**

#### **See also:**

*det*

### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

The determinant is computed via LU factorization using the LAPACK routine z/dgetrf.

### **Examples**

The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

```
>>> import numpy as np
>>> a = np.array([[1, 2], [3, 4]])
>>> (sign, logabsdet) = np.linalg.slogdet(a)
>>> (sign, logabsdet)
(-1, 0.69314718055994529) # may vary
>>> sign * np.exp(logabsdet)
-2.0
```
Computing log-determinants for a stack of matrices:

```
>>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
>>> a.shape
(3, 2, 2)
>>> sign, logabsdet = np.linalg.slogdet(a)
>>> (sign, logabsdet)
(array([-1., -1., -1.]), array([ 0.69314718, 1.09861229, 2.07944154]))
>>> sign * np.exp(logabsdet)
array([-2., -3., -8.])
```
This routine succeeds where ordinary *det* does not:

```
>>> np.linalg.det(np.eye(500) * 0.1)
0.0
>>> np.linalg.slogdet(np.eye(500) * 0.1)
(1, -1151.2925464970228)
```
numpy.**trace**(*a*, *offset=0*, *axis1=0*, *axis2=1*, *dtype=None*, *out=None*)

Return the sum along diagonals of the array.

If *a* is 2-D, the sum along its diagonal with the given offset is returned, i.e., the sum of elements a[i,i+offset] for all i.

If *a* has more than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D subarrays whose traces are returned. The shape of the resulting array is the same as that of *a* with *axis1* and *axis2* removed.

### **Parameters**

**a**

[array_like] Input array, from which the diagonals are taken.

#### **offset**

[int, optional] Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.

- **axis1, axis2**
[int, optional] Axes to be used as the first and second axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults are the first two axes of *a*.

#### **dtype**

[dtype, optional] Determines the data-type of the returned array and of the accumulator where the elements are summed. If dtype has the value None and *a* is of integer type of precision less than the default integer precision, then the default integer precision is used. Otherwise, the precision is the same as that of *a*.

### **out**

[ndarray, optional] Array into which the output is placed. Its type is preserved and it must be of the right shape to hold the output.

#### **Returns**

#### **sum_along_diagonals**

[ndarray] If *a* is 2-D, the sum along the diagonal is returned. If *a* has larger dimensions, then an array of sums along diagonals is returned.

#### **See also:**

*diag***,** *diagonal***,** *diagflat*

#### **Examples**

```
>>> import numpy as np
>>> np.trace(np.eye(3))
3.0
>>> a = np.arange(8).reshape((2,2,2))
>>> np.trace(a)
array([6, 8])
```

```
>>> a = np.arange(24).reshape((2,2,2,3))
>>> np.trace(a).shape
(2, 3)
```
#### linalg.**trace**(*x*, */* , ***, *offset=0*, *dtype=None*)

Returns the sum along the specified diagonals of a matrix (or a stack of matrices) x.

This function is Array API compatible, contrary to *numpy.trace*.

#### **Parameters**

#### **x**

[(…,M,N) array_like] Input array having shape (…, M, N) and whose innermost two dimensions form MxN matrices.

#### **offset**

[int, optional] Offset specifying the off-diagonal relative to the main diagonal, where:

```
* offset = 0: the main diagonal.
* offset > 0: off-diagonal above the main diagonal.
* offset < 0: off-diagonal below the main diagonal.
```
#### **dtype**

[dtype, optional] Data type of the returned array.

#### **Returns**

#### **out**

[ndarray] An array containing the traces and whose shape is determined by removing the last two dimensions and storing the traces in the last array dimension. For example, if x has rank k and shape: (I, J, K, …, L, M, N), then an output array has rank k-2 and shape: (I, J, K, …, L) where:

```
out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])
```
The returned array must have a data type as described by the dtype parameter above.

**See also:**

*numpy.trace*

#### **Examples**

```
>>> np.linalg.trace(np.eye(3))
3.0
>>> a = np.arange(8).reshape((2, 2, 2))
>>> np.linalg.trace(a)
array([3, 11])
```
Trace is computed with the last two axes as the 2-d sub-arrays. This behavior differs from *numpy.trace* which uses the first two axes by default.

```
>>> a = np.arange(24).reshape((3, 2, 2, 2))
>>> np.linalg.trace(a).shape
(3, 2)
```
Traces adjacent to the main diagonal can be obtained by using the *offset* argument:

```
>>> a = np.arange(9).reshape((3, 3)); a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> np.linalg.trace(a, offset=1) # First superdiagonal
6
>>> np.linalg.trace(a, offset=2) # Second superdiagonal
2
>>> np.linalg.trace(a, offset=-1) # First subdiagonal
10
>>> np.linalg.trace(a, offset=-2) # Second subdiagonal
6
```
#### **Solving equations and inverting matrices**

| linalg.solve(a, b) | Solve a linear matrix equation, or system of linear scalar |
| --- | --- |
|  | equations. |
| linalg.tensorsolve(a, b[, axes]) | Solve the tensor equation a x = b for x. |
| linalg.lstsq(a, b[, rcond]) | Return the least-squares solution to a linear matrix equa |
|  | tion. |
| linalg.inv(a) | Compute the inverse of a matrix. |
| linalg.pinv(a[, rcond, hermitian, rtol]) | Compute the (Moore-Penrose) pseudo-inverse of a ma |
|  | trix. |
| linalg.tensorinv(a[, ind]) | Compute the 'inverse' of an N-dimensional array. |

#### linalg.**solve**(*a*, *b*)

Solve a linear matrix equation, or system of linear scalar equations.

Computes the "exact" solution, *x*, of the well-determined, i.e., full rank, linear matrix equation *ax = b*.

#### **Parameters**

**a**

[(…, M, M) array_like] Coefficient matrix.

**b**

[{(M,), (…, M, K)}, array_like] Ordinate or "dependent variable" values.

#### **Returns**

```
x
```
[{(…, M,), (…, M, K)} ndarray] Solution to the system a x = b. Returned shape is (…, M) if b is shape (M,) and (…, M, K) if b is (…, M, K), where the "…" part is broadcasted between a and b.

#### **Raises**

**LinAlgError** If *a* is singular or not square.

#### **See also:**

**scipy.linalg.solve**

Similar function in SciPy.

#### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

The solutions are computed using LAPACK routine _gesv.

*a* must be square and of full-rank, i.e., all rows (or, equivalently, columns) must be linearly independent; if either is not true, use *lstsq* for the least-squares best "solution" of the system/equation.

Changed in version 2.0: The b array is only treated as a shape (M,) column vector if it is exactly 1-dimensional. In all other instances it is treated as a stack of (M, K) matrices. Previously b would be treated as a stack of (M,) vectors if b.ndim was equal to a.ndim - 1.

#### **References**

[1]

### **Examples**

Solve the system of equations: x0 + 2 * x1 = 1 and 3 * x0 + 5 * x1 = 2:

```
>>> import numpy as np
>>> a = np.array([[1, 2], [3, 5]])
>>> b = np.array([1, 2])
>>> x = np.linalg.solve(a, b)
>>> x
array([-1., 1.])
```
Check that the solution is correct:

```
>>> np.allclose(np.dot(a, x), b)
True
```
### linalg.**tensorsolve**(*a*, *b*, *axes=None*)

Solve the tensor equation a x = b for x.

It is assumed that all indices of *x* are summed over in the product, together with the rightmost indices of *a*, as is done in, for example, tensordot(a, x, axes=x.ndim).

#### **Parameters**

#### **a**

[array_like] Coefficient tensor, of shape b.shape + Q. *Q*, a tuple, equals the shape of that sub-tensor of *a* consisting of the appropriate number of its rightmost indices, and must be such that prod(Q) == prod(b.shape) (in which sense *a* is said to be 'square').

#### **b**

[array_like] Right-hand tensor, which can be of any shape.

#### **axes**

[tuple of ints, optional] Axes in *a* to reorder to the right, before inversion. If None (default), no reordering is done.

#### **Returns**

**x**

[ndarray, shape Q]

### **Raises**

### **LinAlgError**

If *a* is singular or not 'square' (in the above sense).

### **See also:**

*numpy.tensordot***,** *tensorinv***,** *numpy.einsum*

### **Examples**

```
>>> import numpy as np
>>> a = np.eye(2*3*4)
>>> a.shape = (2*3, 4, 2, 3, 4)
>>> rng = np.random.default_rng()
>>> b = rng.normal(size=(2*3, 4))
>>> x = np.linalg.tensorsolve(a, b)
>>> x.shape
(2, 3, 4)
>>> np.allclose(np.tensordot(a, x, axes=3), b)
True
```
linalg.**lstsq**(*a*, *b*, *rcond=None*)

Return the least-squares solution to a linear matrix equation.

Computes the vector *x* that approximately solves the equation a @ x = b. The equation may be under-, well-, or over-determined (i.e., the number of linearly independent rows of *a* can be less than, equal to, or greater than its number of linearly independent columns). If *a* is square and of full rank, then *x* (but for round-off error) is the "exact" solution of the equation. Else, *x* minimizes the Euclidean 2-norm *||b − ax||*. If there are multiple minimizing solutions, the one with the smallest 2-norm *||x||* is returned.

#### **Parameters**

**a**

[(M, N) array_like] "Coefficient" matrix.

**b**

[{(M,), (M, K)} array_like] Ordinate or "dependent variable" values. If *b* is two-dimensional, the least-squares solution is calculated for each of the *K* columns of *b*.

### **rcond**

[float, optional] Cut-off ratio for small singular values of *a*. For the purposes of rank determination, singular values are treated as zero if they are smaller than *rcond* times the largest singular value of *a*. The default uses the machine precision times max(M, N). Passing -1 will use machine precision.

Changed in version 2.0: Previously, the default was -1, but a warning was given that this would change.

#### **Returns**

**x**

[{(N,), (N, K)} ndarray] Least-squares solution. If *b* is two-dimensional, the solutions are in the *K* columns of *x*.

#### **residuals**

[{(1,), (K,), (0,)} ndarray] Sums of squared residuals: Squared Euclidean 2-norm for each column in b - a @ x. If the rank of *a* is < N or M <= N, this is an empty array. If *b* is 1-dimensional, this is a (1,) shape array. Otherwise the shape is (K,).

### **rank**

[int] Rank of matrix *a*.

### **s**

[(min(M, N),) ndarray] Singular values of *a*.

#### **Raises**

**LinAlgError**

If computation does not converge.

#### **See also:**

```
scipy.linalg.lstsq
    Similar function in SciPy.
```
#### **Notes**

If *b* is a matrix, then all array results are returned as matrices.

#### **Examples**

Fit a line, y = mx + c, through some noisy data-points:

**>>> import numpy as np >>>** x = np.array([0, 1, 2, 3]) **>>>** y = np.array([-1, 0.2, 0.9, 2.1])

By examining the coefficients, we see that the line should have a gradient of roughly 1 and cut the y-axis at, more or less, -1.

We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]]. Now use *lstsq* to solve for *p*:

```
>>> A = np.vstack([x, np.ones(len(x))]).T
>>> A
array([[ 0., 1.],
       [ 1., 1.],
       [ 2., 1.],
       [ 3., 1.]])
```

```
>>> m, c = np.linalg.lstsq(A, y)[0]
>>> m, c
(1.0 -0.95) # may vary
```
Plot the data along with the fitted line:

```
>>> import matplotlib.pyplot as plt
>>> _ = plt.plot(x, y, 'o', label='Original data', markersize=10)
>>> _ = plt.plot(x, m*x + c, 'r', label='Fitted line')
>>> _ = plt.legend()
>>> plt.show()
```
![](_page_101_Figure_10.jpeg)

#### linalg.**inv**(*a*)

Compute the inverse of a matrix.

Given a square matrix *a*, return the matrix *ainv* satisfying a @ ainv = ainv @ a = eye(a.shape[0]).

**Parameters**

**a**

[(…, M, M) array_like] Matrix to be inverted.

**Returns**

**ainv**

[(…, M, M) ndarray or matrix] Inverse of the matrix *a*.

**Raises**

**LinAlgError**

If *a* is not square or inversion fails.

**See also:**

**scipy.linalg.inv** Similar function in SciPy.

*numpy.linalg.cond* Compute the condition number of a matrix.

*numpy.linalg.svd*

Compute the singular value decomposition of a matrix.

### **Notes**

Broadcasting rules apply, see the *numpy.linalg* documentation for details.

If *a* is detected to be singular, a *LinAlgError* is raised. If *a* is ill-conditioned, a *LinAlgError* may or may not be raised, and results may be inaccurate due to floating-point errors.

#### **References**

[1]

### **Examples**

```
>>> import numpy as np
>>> from numpy.linalg import inv
>>> a = np.array([[1., 2.], [3., 4.]])
>>> ainv = inv(a)
>>> np.allclose(a @ ainv, np.eye(2))
True
>>> np.allclose(ainv @ a, np.eye(2))
True
```
If a is a matrix object, then the return value is a matrix as well:

```
>>> ainv = inv(np.matrix(a))
>>> ainv
matrix([[-2. , 1. ],
        [ 1.5, -0.5]])
```
Inverses of several matrices can be computed at once:

```
>>> a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
>>> inv(a)
array([[[-2. , 1. ],
        [ 1.5 , -0.5 ]],
       [[-1.25, 0.75],
        [ 0.75, -0.25]]])
```
If a matrix is close to singular, the computed inverse may not satisfy a @ ainv = ainv @ a = eye(a. shape[0]) even if a *LinAlgError* is not raised:

```
>>> a = np.array([[2,4,6],[2,0,2],[6,8,14]])
>>> inv(a) # No errors raised
array([[-1.12589991e+15, -5.62949953e+14, 5.62949953e+14],
  [-1.12589991e+15, -5.62949953e+14, 5.62949953e+14],
  [ 1.12589991e+15, 5.62949953e+14, -5.62949953e+14]])
>>> a @ inv(a)
array([[ 0. , -0.5 , 0. ], # may vary
      [-0.5 , 0.625, 0.25 ],
      [ 0. , 0. , 1. ]])
```
To detect ill-conditioned matrices, you can use *numpy.linalg.cond* to compute its *condition number* [1]. The larger the condition number, the more ill-conditioned the matrix is. As a rule of thumb, if the condition number cond(a) = 10**k, then you may lose up to k digits of accuracy on top of what would be lost to the numerical method due to loss of precision from arithmetic methods.

```
>>> from numpy.linalg import cond
>>> cond(a)
np.float64(8.659885634118668e+17) # may vary
```
It is also possible to detect ill-conditioning by inspecting the matrix's singular values directly. The ratio between the largest and the smallest singular value is the condition number:

```
>>> from numpy.linalg import svd
>>> sigma = svd(a, compute_uv=False) # Do not compute singular vectors
>>> sigma.max()/sigma.min()
8.659885634118668e+17 # may vary
```
linalg.**pinv**(*a*, *rcond=None*, *hermitian=False*, ***, *rtol=<no value>*)

Compute the (Moore-Penrose) pseudo-inverse of a matrix.

Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) and including all *large* singular values.

### **Parameters**

**a**

[(…, M, N) array_like] Matrix or stack of matrices to be pseudo-inverted.

#### **rcond**

[(…) array_like of float, optional] Cutoff for small singular values. Singular values less than or equal to rcond * largest_singular_value are set to zero. Broadcasts against the stack of matrices. Default: 1e-15.

#### **hermitian**

[bool, optional] If True, *a* is assumed to be Hermitian (symmetric if real-valued), enabling a more efficient method for finding singular values. Defaults to False.

#### **rtol**

[(…) array_like of float, optional] Same as *rcond*, but it's an Array API compatible parameter

name. Only *rcond* or *rtol* can be set at a time. If none of them are provided then NumPy's 1e-15 default is used. If rtol=None is passed then the API standard default is used.

New in version 2.0.0.

#### **Returns**

**B**

[(…, N, M) ndarray] The pseudo-inverse of *a*. If *a* is a *matrix* instance, then so is *B*.

**Raises**

**LinAlgError**

If the SVD computation does not converge.

**See also:**

```
scipy.linalg.pinv
```
Similar function in SciPy.

```
scipy.linalg.pinvh
```
Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

#### **Notes**

The pseudo-inverse of a matrix A, denoted *A*+, is defined as: "the matrix that 'solves' [the least-squares problem] *Ax* = *b*," i.e., if *x*¯ is said solution, then *A*+ is that matrix such that *x*¯ = *A*+*b*.

It can be shown that if *Q*1Σ*QT* 2 = *A* is the singular value decomposition of A, then *A*+ = *Q*2Σ +*QT* 1 , where *Q*1*,*2 are orthogonal matrices, Σ is a diagonal matrix consisting of A's so-called singular values, (followed, typically, by zeros), and then Σ + is simply the diagonal matrix consisting of the reciprocals of A's singular values (again, followed by zeros). [1]

#### **References**

[1]

#### **Examples**

The following example checks that a * a+ * a == a and a+ * a * a+ == a+:

```
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> a = rng.normal(size=(9, 6))
>>> B = np.linalg.pinv(a)
>>> np.allclose(a, np.dot(a, np.dot(B, a)))
True
>>> np.allclose(B, np.dot(B, np.dot(a, B)))
True
```
### linalg.**tensorinv**(*a*, *ind=2*)

Compute the 'inverse' of an N-dimensional array.

The result is an inverse for *a* relative to the tensordot operation tensordot(a, b, ind), i. e., up to floatingpoint accuracy, tensordot(tensorinv(a), a, ind) is the "identity" tensor for the tensordot operation.

**Parameters**

### **a**

```
[array_like] Tensor to 'invert'. Its shape must be 'square', i. e., prod(a.shape[:ind])
== prod(a.shape[ind:]).
```
#### **ind**

[int, optional] Number of first indices that are involved in the inverse sum. Must be a positive integer, default is 2.

### **Returns**

**b**

```
[ndarray] a's tensordot inverse, shape a.shape[ind:] + a.shape[:ind].
```
### **Raises**

### **LinAlgError**

If *a* is singular or not 'square' (in the above sense).

### **See also:**

*numpy.tensordot***,** *tensorsolve*

### **Examples**

```
>>> import numpy as np
>>> a = np.eye(4*6)
>>> a.shape = (4, 6, 8, 3)
>>> ainv = np.linalg.tensorinv(a, ind=2)
>>> ainv.shape
(8, 3, 4, 6)
>>> rng = np.random.default_rng()
>>> b = rng.normal(size=(4, 6))
>>> np.allclose(np.tensordot(ainv, b), np.linalg.tensorsolve(a, b))
True
```

```
>>> a = np.eye(4*6)
>>> a.shape = (24, 8, 3)
>>> ainv = np.linalg.tensorinv(a, ind=1)
>>> ainv.shape
(8, 3, 24)
>>> rng = np.random.default_rng()
>>> b = rng.normal(size=24)
>>> np.allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))
True
```
#### **Other matrix operations**

| diagonal(a[, offset, axis1, axis2]) | Return specified diagonals. |
| --- | --- |
| linalg.diagonal(x, /, *[, offset]) | Returns specified diagonals of a matrix (or a stack of ma |
|  | trices) x. |
| linalg.matrix_transpose(x, /) | Transposes a matrix (or a stack of matrices) x. |

numpy.**diagonal**(*a*, *offset=0*, *axis1=0*, *axis2=1*)

Return specified diagonals.

If *a* is 2-D, returns the diagonal of *a* with the given offset, i.e., the collection of elements of the form a[i, i+offset]. If *a* has more than two dimensions, then the axes specified by *axis1* and *axis2* are used to determine the 2-D sub-array whose diagonal is returned. The shape of the resulting array can be determined by removing *axis1* and *axis2* and appending an index to the right equal to the size of the resulting diagonals.

In versions of NumPy prior to 1.7, this function always returned a new, independent array containing a copy of the values in the diagonal.

In NumPy 1.7 and 1.8, it continues to return a copy of the diagonal, but depending on this fact is deprecated. Writing to the resulting array continues to work as it used to, but a FutureWarning is issued.

Starting in NumPy 1.9 it returns a read-only view on the original array. Attempting to write to the resulting array will produce an error.

In some future release, it will return a read/write view and writing to the returned array will alter your original array. The returned array will have the same type as the input array.

If you don't write to the array returned by this function, then you can just ignore all of the above.

If you depend on the current behavior, then we suggest copying the returned array explicitly, i.e., use np. diagonal(a).copy() instead of just np.diagonal(a). This will work with both past and future versions of NumPy.

#### **Parameters**

#### **a**

[array_like] Array from which the diagonals are taken.

#### **offset**

[int, optional] Offset of the diagonal from the main diagonal. Can be positive or negative. Defaults to main diagonal (0).

#### **axis1**

[int, optional] Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to first axis (0).

#### **axis2**

[int, optional] Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to second axis (1).

#### **Returns**

#### **array_of_diagonals**

[ndarray] If *a* is 2-D, then a 1-D array containing the diagonal and of the same type as *a* is returned unless *a* is a *matrix*, in which case a 1-D array rather than a (2-D) *matrix* is returned in order to maintain backward compatibility.

If a.ndim > 2, then the dimensions specified by *axis1* and *axis2* are removed, and a new axis inserted at the end corresponding to the diagonal.

### **Raises**

#### **ValueError**

If the dimension of *a* is less than 2.

### **See also:**

*diag* MATLAB work-a-like for 1-D and 2-D arrays.

### *diagflat*

Create diagonal arrays.

#### *trace*

Sum along diagonals.

### **Examples**

```
>>> import numpy as np
>>> a = np.arange(4).reshape(2,2)
>>> a
array([[0, 1],
       [2, 3]])
>>> a.diagonal()
array([0, 3])
>>> a.diagonal(1)
array([1])
```
A 3-D example:

```
>>> a = np.arange(8).reshape(2,2,2); a
array([[[0, 1],
       [2, 3]],
      [[4, 5],
       [6, 7]]])
>>> a.diagonal(0, # Main diagonals of two arrays created by skipping
... 0, # across the outer(left)-most axis last and
... 1) # the "middle" (row) axis first.
array([[0, 6],
      [1, 7]])
```
The sub-arrays whose main diagonals we just obtained; note that each corresponds to fixing the right-most (column) axis, and that the diagonals are "packed" in rows.

```
>>> a[:,:,0] # main diagonal is [0 6]
array([[0, 2],
       [4, 6]])
>>> a[:,:,1] # main diagonal is [1 7]
array([[1, 3],
       [5, 7]])
```
The anti-diagonal can be obtained by reversing the order of elements using either *numpy.flipud* or *numpy. fliplr*.

```
>>> a = np.arange(9).reshape(3, 3)
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> np.fliplr(a).diagonal() # Horizontal flip
array([2, 4, 6])
>>> np.flipud(a).diagonal() # Vertical flip
array([6, 4, 2])
```
Note that the order in which the diagonal is retrieved varies depending on the flip function.

```
linalg.diagonal(x, / , *, offset=0)
```
Returns specified diagonals of a matrix (or a stack of matrices) x.

This function is Array API compatible, contrary to *numpy.diagonal*, the matrix is assumed to be defined by the last two dimensions.

#### **Parameters**

#### **x**

[(…,M,N) array_like] Input array having shape (…, M, N) and whose innermost two dimensions form MxN matrices.

#### **offset**

[int, optional] Offset specifying the off-diagonal relative to the main diagonal, where:

```
* offset = 0: the main diagonal.
* offset > 0: off-diagonal above the main diagonal.
* offset < 0: off-diagonal below the main diagonal.
```
#### **Returns**

#### **out**

[(…,min(N,M)) ndarray] An array containing the diagonals and whose shape is determined by removing the last two dimensions and appending a dimension equal to the size of the resulting diagonals. The returned array must have the same data type as x.

### **See also:**

#### *numpy.diagonal*

#### **Examples**

```
>>> a = np.arange(4).reshape(2, 2); a
array([[0, 1],
       [2, 3]])
>>> np.linalg.diagonal(a)
array([0, 3])
```
#### A 3-D example:

```
>>> a = np.arange(8).reshape(2, 2, 2); a
array([[[0, 1],
        [2, 3]],
       [[4, 5],
        [6, 7]]])
>>> np.linalg.diagonal(a)
array([[0, 3],
       [4, 7]])
```
Diagonals adjacent to the main diagonal can be obtained by using the *offset* argument:

```
>>> a = np.arange(9).reshape(3, 3)
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> np.linalg.diagonal(a, offset=1) # First superdiagonal
array([1, 5])
>>> np.linalg.diagonal(a, offset=2) # Second superdiagonal
array([2])
```
(continues on next page)

```
>>> np.linalg.diagonal(a, offset=-1) # First subdiagonal
array([3, 7])
>>> np.linalg.diagonal(a, offset=-2) # Second subdiagonal
array([6])
```
The anti-diagonal can be obtained by reversing the order of elements using either *numpy.flipud* or *numpy. fliplr*.

```
>>> a = np.arange(9).reshape(3, 3)
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> np.linalg.diagonal(np.fliplr(a)) # Horizontal flip
array([2, 4, 6])
>>> np.linalg.diagonal(np.flipud(a)) # Vertical flip
array([6, 4, 2])
```
Note that the order in which the diagonal is retrieved varies depending on the flip function.

#### linalg.**matrix_transpose**(*x*, */* )

Transposes a matrix (or a stack of matrices) x.

This function is Array API compatible.

#### **Parameters**

**x**

[array_like] Input array having shape (…, M, N) and whose two innermost dimensions form MxN matrices.

#### **Returns**

#### **out**

[ndarray] An array containing the transpose for each matrix and having shape (…, N, M).

### **See also:**

#### *transpose*

Generic transpose method.

#### **Examples**

```
>>> import numpy as np
>>> np.matrix_transpose([[1, 2], [3, 4]])
array([[1, 3],
       [2, 4]])
```

```
>>> np.matrix_transpose([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
array([[[1, 3],
        [2, 4]],
       [[5, 7],
        [6, 8]]])
```
(continued from previous page)

### **Exceptions**

| linalg.LinAlgError | Generic Python-exception-derived object raised by linalg |
| --- | --- |
|  | functions. |

### **exception** linalg.**LinAlgError**

Generic Python-exception-derived object raised by linalg functions.

General purpose exception class, derived from Python's ValueError class, programmatically raised in linalg functions when a Linear Algebra-related condition would prevent further correct execution of the function.

**Parameters**

**None**

### **Examples**

```
>>> from numpy import linalg as LA
>>> LA.inv(np.zeros((2,2)))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "...linalg.py", line 350,
    in inv return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))
  File "...linalg.py", line 249,
    in solve
    raise LinAlgError('Singular matrix')
numpy.linalg.LinAlgError: Singular matrix
```
### **Linear algebra on several matrices at once**

Several of the linear algebra routines listed above are able to compute results for several matrices at once, if they are stacked into the same array.

This is indicated in the documentation via input parameter specifications such as a : (..., M, M) array_like. This means that if for instance given an input array a.shape == (N, M, M), it is interpreted as a "stack" of N matrices, each of size M-by-M. Similar specification applies to return values, for instance the determinant has det : (...) and will in this case return an array of shape det(a).shape == (N,). This generalizes to linear algebra operations on higher-dimensional arrays: the last 1 or 2 dimensions of a multidimensional array are interpreted as vectors or matrices, as appropriate for each operation.

### **numpy.polynomial**

A sub-package for efficiently dealing with polynomials.

Within the documentation for this sub-package, a "finite power series," i.e., a polynomial (also referred to simply as a "series") is represented by a 1-D numpy array of the polynomial's coefficients, ordered from lowest order term to highest. For example, array([1,2,3]) represents P_0 + 2*P_1 + 3*P_2, where P_n is the n-th order basis polynomial applicable to the specific module in question, e.g., *polynomial* (which "wraps" the "standard" basis) or *chebyshev*. For optimal performance, all operations on polynomials, including evaluation at an argument, are implemented as operations on the coefficients. Additional (module-specific) information can be found in the docstring for the module of interest.

This package provides *convenience classes* for each of six different kinds of polynomials:

| Name | Provides |
| --- | --- |
| Polynomial | Power series |
| Chebyshev | Chebyshev series |
| Legendre | Legendre series |
| Laguerre | Laguerre series |
| Hermite | Hermite series |
| HermiteE | HermiteE series |

These *convenience classes* provide a consistent interface for creating, manipulating, and fitting data with polynomials of different bases. The convenience classes are the preferred interface for the *polynomial* package, and are available from the numpy.polynomial namespace. This eliminates the need to navigate to the corresponding submodules, e.g. np. polynomial.Polynomial or np.polynomial.Chebyshev instead of np.polynomial.polynomial. Polynomial or np.polynomial.chebyshev.Chebyshev, respectively. The classes provide a more consistent and concise interface than the type-specific functions defined in the submodules for each type of polynomial. For example, to fit a Chebyshev polynomial with degree 1 to data given by arrays xdata and ydata, the *fit* class method:

```
>>> from numpy.polynomial import Chebyshev
>>> xdata = [1, 2, 3, 4]
>>> ydata = [1, 4, 9, 16]
>>> c = Chebyshev.fit(xdata, ydata, deg=1)
```
is preferred over the *chebyshev.chebfit* function from the np.polynomial.chebyshev module:

```
>>> from numpy.polynomial.chebyshev import chebfit
>>> c = chebfit(xdata, ydata, deg=1)
```
See *Using the convenience classes* for more details.

### **Convenience Classes**

The following lists the various constants and methods common to all of the classes representing the various kinds of polynomials. In the following, the term Poly represents any one of the convenience classes (e.g. *Polynomial*, *Chebyshev*, *Hermite*, etc.) while the lowercase p represents an **instance** of a polynomial class.

### **Constants**

- Poly.domain Default domain
- Poly.window Default window
- Poly.basis_name String used to represent the basis
- Poly.maxpower Maximum value n such that p**n is allowed
- Poly.nickname String used in printing

### **Creation**

Methods for creating polynomial instances.

- Poly.basis(degree) Basis polynomial of given degree
- Poly.identity() p where p(x) = x for all x
- Poly.fit(x, y, deg) p of degree deg with coefficients determined by the least-squares fit to the data x, y
- Poly.fromroots(roots) p with specified roots
- p.copy() Create a copy of p

### **Conversion**

Methods for converting a polynomial instance of one kind to another.

- p.cast(Poly) Convert p to instance of kind Poly
- p.convert(Poly) Convert p to instance of kind Poly or map between domain and window

### **Calculus**

- p.deriv() Take the derivative of p
- p.integ() Integrate p

### **Validation**

- Poly.has_samecoef(p1, p2) Check if coefficients match
- Poly.has_samedomain(p1, p2) Check if domains match
- Poly.has_sametype(p1, p2) Check if types match
- Poly.has_samewindow(p1, p2) Check if windows match

#### **Misc**

- p.linspace() Return x, p(x) at equally-spaced points in domain
- p.mapparms() Return the parameters for the linear mapping between domain and window.
- p.roots() Return the roots of p.
- p.trim() Remove trailing coefficients.
- p.cutdeg(degree) Truncate p to given degree
- p.truncate(size) Truncate p to given size

### **Configuration**

| numpy.polynomial. | Set the default format for the string representation of poly |
| --- | --- |
| set_default_printstyle(style) | nomials. |

### polynomial.**set_default_printstyle**(*style*)

Set the default format for the string representation of polynomials.

Values for style must be valid inputs to __format__, i.e. 'ascii' or 'unicode'.

#### **Parameters**

### **style**

[str] Format string for default printing style. Must be either 'ascii' or 'unicode'.

### **Notes**

The default format depends on the platform: 'unicode' is used on Unix-based systems and 'ascii' on Windows. This determination is based on default font support for the unicode superscript and subscript ranges.

### **Examples**

```
>>> p = np.polynomial.Polynomial([1, 2, 3])
>>> c = np.polynomial.Chebyshev([1, 2, 3])
>>> np.polynomial.set_default_printstyle('unicode')
>>> print(p)
1.0 + 2.0·x + 3.0·x²
>>> print(c)
1.0 + 2.0·T₁(x) + 3.0·T₂(x)
>>> np.polynomial.set_default_printstyle('ascii')
>>> print(p)
1.0 + 2.0 x + 3.0 x**2
>>> print(c)
1.0 + 2.0 T_1(x) + 3.0 T_2(x)
>>> # Formatting supersedes all class/package-level defaults
>>> print(f"{p:unicode}")
1.0 + 2.0·x + 3.0·x²
```
### **Random sampling (numpy.random)**

#### **Quick start**

The *numpy.random* module implements pseudo-random number generators (PRNGs or RNGs, for short) with the ability to draw samples from a variety of probability distributions. In general, users will create a *Generator* instance with *default_rng* and call the various methods on it to obtain samples from different distributions.

```
>>> import numpy as np
>>> rng = np.random.default_rng()
# Generate one random float uniformly distributed over the range [0, 1)
>>> rng.random()
0.06369197489564249 # may vary
# Generate an array of 10 numbers according to a unit Gaussian distribution
>>> rng.standard_normal(10)
```
(continues on next page)

(continued from previous page)

```
array([-0.31018314, -1.8922078 , -0.3628523 , -0.63526532, 0.43181166, # may vary
       0.51640373, 1.25693945, 0.07779185, 0.84090247, -2.13406828])
# Generate an array of 5 integers uniformly over the range [0, 10)
>>> rng.integers(low=0, high=10, size=5)
array([8, 7, 6, 2, 0]) # may vary
```
Our RNGs are deterministic sequences and can be reproduced by specifying a seed integer to derive its initial state. By default, with no seed provided, *default_rng* will seed the RNG from nondeterministic data from the operating system and therefore generate different numbers each time. The pseudo-random sequences will be independent for all practical purposes, at least those purposes for which our pseudo-randomness was good for in the first place.

```
>>> import numpy as np
>>> rng1 = np.random.default_rng()
>>> rng1.random()
0.6596288841243357 # may vary
>>> rng2 = np.random.default_rng()
>>> rng2.random()
0.11885628817151628 # may vary
```
**Warning:** The pseudo-random number generators implemented in this module are designed for statistical modeling and simulation. They are not suitable for security or cryptographic purposes. See the secrets module from the standard library for such use cases.

Seeds should be large positive integers. *default_rng* can take positive integers of any size. We recommend using very large, unique numbers to ensure that your seed is different from anyone else's. This is good practice to ensure that your results are statistically independent from theirs unless you are intentionally *trying* to reproduce their result. A convenient way to get such a seed number is to use secrets.randbits to get an arbitrary 128-bit integer.

```
>>> import numpy as np
>>> import secrets
>>> secrets.randbits(128)
122807528840384100672342137672332424406 # may vary
>>> rng1 = np.random.default_rng(122807528840384100672342137672332424406)
>>> rng1.random()
0.5363922081269535
>>> rng2 = np.random.default_rng(122807528840384100672342137672332424406)
>>> rng2.random()
0.5363922081269535
```
See the documentation on *default_rng* and *SeedSequence* for more advanced options for controlling the seed in specialized scenarios.

*Generator* and its associated infrastructure was introduced in NumPy version 1.17.0. There is still a lot of code that uses the older *RandomState* and the functions in *numpy.random*. While there are no plans to remove them at this time, we do recommend transitioning to *Generator* as you can. The algorithms are faster, more flexible, and will receive more improvements in the future. For the most part, *Generator* can be used as a replacement for *RandomState*. See *Legacy random generation* for information on the legacy infrastructure, *What's new or different* for information on transitioning, and NEP 19 for some of the reasoning for the transition.

### **Design**

Users primarily interact with *Generator* instances. Each *Generator* instance owns a *BitGenerator* instance that implements the core RNG algorithm. The *BitGenerator* has a limited set of responsibilities. It manages state and provides functions to produce random doubles and random unsigned 32- and 64-bit values.

The *Generator* takes the bit generator-provided stream and transforms them into more useful distributions, e.g., simulated normal random values. This structure allows alternative bit generators to be used with little code duplication.

NumPy implements several different *BitGenerator* classes implementing different RNG algorithms. *default_rng* currently uses *PCG64* as the default *BitGenerator*. It has better statistical properties and performance than the *MT19937* algorithm used in the legacy *RandomState*. See *Bit generators* for more details on the supported BitGenerators.

*default_rng* and BitGenerators delegate the conversion of seeds into RNG states to *SeedSequence* internally. *SeedSequence* implements a sophisticated algorithm that intermediates between the user's input and the internal implementation details of each *BitGenerator* algorithm, each of which can require different amounts of bits for its state. Importantly, it lets you use arbitrary-sized integers and arbitrary sequences of such integers to mix together into the RNG state. This is a useful primitive for constructing a *flexible pattern for parallel RNG streams*.

For backward compatibility, we still maintain the legacy *RandomState* class. It continues to use the *MT19937* algorithm by default, and old seeds continue to reproduce the same results. The convenience *Functions in numpy.random* are still aliases to the methods on a single global *RandomState* instance. See *Legacy random generation* for the complete details. See *What's new or different* for a detailed comparison between *Generator* and *RandomState*.

### **Parallel Generation**

The included generators can be used in parallel, distributed applications in a number of ways:

- *SeedSequence spawning*
- *Sequence of integer seeds*
- *Independent streams*
- *Jumping the BitGenerator state*

Users with a very large amount of parallelism will want to consult *Upgrading PCG64 with PCG64DXSM*.

### **Concepts**

### **Random Generator**

The *Generator* provides access to a wide range of distributions, and served as a replacement for *RandomState*. The main difference between the two is that *Generator* relies on an additional BitGenerator to manage state and generate the random bits, which are then transformed into random values from useful distributions. The default BitGenerator used by *Generator* is *PCG64*. The BitGenerator can be changed by passing an instantized BitGenerator to *Generator*.

numpy.random.**default_rng**(*seed=None*)

Construct a new Generator with the default BitGenerator (PCG64).

### **Parameters**

### **seed**

[{None, int, array_like[ints], SeedSequence, BitGenerator, Generator, RandomState}, optional] A seed to initialize the *BitGenerator*. If None, then fresh, unpredictable entropy will be pulled from the OS. If an int or array_like[ints] is passed, then all values must be non-negative and will be passed to *SeedSequence* to derive the initial *BitGenerator* state. One may also pass in a *SeedSequence* instance. Additionally, when passed a *BitGenerator*, it will be wrapped by *Generator*. If passed a *Generator*, it will be returned unaltered. When passed a legacy *RandomState* instance it will be coerced to a *Generator*.

#### **Returns**

**Generator**

The initialized generator object.

#### **Notes**

If seed is not a *BitGenerator* or a *Generator*, a new *BitGenerator* is instantiated. This function does not manage a default global instance.

See *Seeding and entropy* for more information about seeding.

#### **Examples**

*default_rng* is the recommended constructor for the random number class *Generator*. Here are several ways we can construct a random number generator using *default_rng* and the *Generator* class.

Here we use *default_rng* to generate a random float:

```
>>> import numpy as np
>>> rng = np.random.default_rng(12345)
>>> print(rng)
Generator(PCG64)
>>> rfloat = rng.random()
>>> rfloat
0.22733602246716966
>>> type(rfloat)
<class 'float'>
```
Here we use *default_rng* to generate 3 random integers between 0 (inclusive) and 10 (exclusive):

```
>>> import numpy as np
>>> rng = np.random.default_rng(12345)
>>> rints = rng.integers(low=0, high=10, size=3)
>>> rints
array([6, 2, 7])
>>> type(rints[0])
<class 'numpy.int64'>
```
Here we specify a seed so that we have reproducible results:

```
>>> import numpy as np
>>> rng = np.random.default_rng(seed=42)
>>> print(rng)
Generator(PCG64)
>>> arr1 = rng.random((3, 3))
>>> arr1
array([[0.77395605, 0.43887844, 0.85859792],
       [0.69736803, 0.09417735, 0.97562235],
       [0.7611397 , 0.78606431, 0.12811363]])
```
If we exit and restart our Python interpreter, we'll see that we generate the same random numbers again:

```
>>> import numpy as np
>>> rng = np.random.default_rng(seed=42)
>>> arr2 = rng.random((3, 3))
>>> arr2
array([[0.77395605, 0.43887844, 0.85859792],
       [0.69736803, 0.09417735, 0.97562235],
       [0.7611397 , 0.78606431, 0.12811363]])
```
**class** numpy.random.**Generator**(*bit_generator*)

Container for the BitGenerators.

*Generator* exposes a number of methods for generating random numbers drawn from a variety of probability distributions. In addition to the distribution-specific arguments, each method takes a keyword argument *size* that defaults to None. If *size* is None, then a single value is generated and returned. If *size* is an integer, then a 1-D array filled with generated values is returned. If *size* is a tuple, then an array with that shape is filled and returned.

The function *numpy.random.default_rng* will instantiate a *Generator* with numpy's default *BitGenerator*.

### **No Compatibility Guarantee**

*Generator* does not provide a version compatibility guarantee. In particular, as better algorithms evolve the bit stream may change.

#### **Parameters**

**bit_generator**

[BitGenerator] BitGenerator to use as the core generator.

#### **See also:**

*default_rng*

Recommended constructor for *Generator*.

### **Notes**

The Python stdlib module random contains pseudo-random number generator with a number of methods that are similar to the ones available in *Generator*. It uses Mersenne Twister, and this bit generator can be accessed using *MT19937*. *Generator*, besides being NumPy-aware, has the advantage that it provides a much larger number of probability distributions to choose from.

### **Examples**

```
>>> from numpy.random import Generator, PCG64
>>> rng = Generator(PCG64())
>>> rng.standard_normal()
-0.203 # random
```
### **Accessing the BitGenerator and spawning**

| bit_generator | Gets the bit generator instance used by the generator |
| --- | --- |
| spawn(n_children) | Create new independent child generators. |

### attribute

random.Generator.**bit_generator**

Gets the bit generator instance used by the generator

#### **Returns**

**bit_generator**

[BitGenerator] The bit generator instance used by the generator

#### method

random.Generator.**spawn**(*n_children*)

Create new independent child generators.

See *SeedSequence spawning* for additional notes on spawning children.

New in version 1.25.0.

#### **Parameters**

**n_children** [int]

**Returns**

**child_generators** [list of Generators]

#### **Raises**

**TypeError**

When the underlying SeedSequence does not implement spawning.

#### **See also:**

*random.BitGenerator.spawn***,** *random.SeedSequence.spawn* Equivalent method on the bit generator and seed sequence.

*bit_generator*

The bit generator instance used by the generator.

### **Examples**

Starting from a seeded default generator:

```
>>> # High quality entropy created with: f"0x{secrets.randbits(128):x}"
>>> entropy = 0x3034c61a9ae04ff8cb62ab8ec2c4b501
>>> rng = np.random.default_rng(entropy)
```
Create two new generators for example for parallel execution:

**>>>** child_rng1, child_rng2 = rng.spawn(2)

Drawn numbers from each are independent but derived from the initial seeding entropy:

**>>>** rng.uniform(), child_rng1.uniform(), child_rng2.uniform() (0.19029263503854454, 0.9475673279178444, 0.4702687338396767)

It is safe to spawn additional children from the original rng or the children:

```
>>> more_child_rngs = rng.spawn(20)
>>> nested_spawn = child_rng1.spawn(20)
```
### **Simple random data**

| integers(low[, high, size, dtype, endpoint]) | Return random integers from low (inclusive) to high (ex |
| --- | --- |
|  | clusive), or if endpoint=True, low (inclusive) to high (in |
|  | clusive). |
| random([size, dtype, out]) | Return random floats in the half-open interval [0.0, 1.0). |
| choice(a[, size, replace, p, axis, shuffle]) | Generates a random sample from a given array |
| bytes(length) | Return random bytes. |

#### method

random.Generator.**integers**(*low*, *high=None*, *size=None*, *dtype=np.int64*, *endpoint=False*)

Return random integers from *low* (inclusive) to *high* (exclusive), or if endpoint=True, *low* (inclusive) to *high* (inclusive). Replaces *RandomState.randint* (with endpoint=False) and *RandomState.random_integers* (with endpoint=True)

Return random integers from the "discrete uniform" distribution of the specified dtype. If *high* is None (the default), then results are from 0 to *low*.

### **Parameters**

#### **low**

[int or array-like of ints] Lowest (signed) integers to be drawn from the distribution (unless high=None, in which case this parameter is 0 and this value is used for *high*).

#### **high**

[int or array-like of ints, optional] If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior if high=None). If array-like, must contain integer values

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

#### **dtype**

[dtype, optional] Desired dtype of the result. Byteorder must be native. The default value is np.int64.

#### **endpoint**

[bool, optional] If true, sample from the interval [low, high] instead of the default [low, high) Defaults to False

#### **Returns**

#### **out**

[int or ndarray of ints] *size*-shaped array of random integers from the appropriate distribution, or a single such random int if *size* not provided.

#### **Notes**

When using broadcasting with uint64 dtypes, the maximum value (2**64) cannot be represented as a standard integer type. The high array (or low if high is None) must have object dtype, e.g., array([2**64]).

### **References**

[1]

#### **Examples**

```
>>> rng = np.random.default_rng()
>>> rng.integers(2, size=10)
array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
>>> rng.integers(1, size=10)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```
Generate a 2 x 4 array of ints between 0 and 4, inclusive:

**>>>** rng.integers(5, size=(2, 4)) array([[4, 0, 2, 1], [3, 2, 2, 0]]) # random

Generate a 1 x 3 array with 3 different upper bounds

**>>>** rng.integers(1, [3, 5, 10]) array([2, 2, 9]) # random

Generate a 1 by 3 array with 3 different lower bounds

```
>>> rng.integers([1, 5, 7], 10)
array([9, 8, 7]) # random
```
Generate a 2 by 4 array using broadcasting with dtype of uint8

```
>>> rng.integers([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
array([[ 8, 6, 9, 7],
      [ 1, 16, 9, 12]], dtype=uint8) # random
```
method

random.Generator.**random**(*size=None*, *dtype=np.float64*, *out=None*)

Return random floats in the half-open interval [0.0, 1.0).

Results are from the "continuous uniform" distribution over the stated interval. To sample *Unif*[*a, b*)*, b > a* use *uniform* or multiply the output of *random* by (b - a) and add a:

(b - a) * random() + a

#### **Parameters**

**size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

### **dtype**

[dtype, optional] Desired dtype of the result, only *float64* and *float32* are supported. Byteorder must be native. The default value is np.float64.

#### **out**

[ndarray, optional] Alternative output array in which to place the result. If size is not None, it must have the same shape as the provided size and must match the type of the output values.

### **Returns**

### **out**

[float or ndarray of floats] Array of random floats of shape *size* (unless size=None, in which case a single float is returned).

### **See also:**

#### *uniform*

Draw samples from the parameterized uniform distribution.

#### **Examples**

```
>>> rng = np.random.default_rng()
>>> rng.random()
0.47108547995356098 # random
>>> type(rng.random())
<class 'float'>
>>> rng.random((5,))
array([ 0.30220482, 0.86820401, 0.1654503 , 0.11659149, 0.54323428]) # random
```
Three-by-two array of random numbers from [-5, 0):

```
>>> 5 * rng.random((3, 2)) - 5
array([[-3.99149989, -0.52338984], # random
       [-2.99091858, -0.79479508],
       [-1.23204345, -1.75224494]])
```
### method

random.Generator.**choice**(*a*, *size=None*, *replace=True*, *p=None*, *axis=0*, *shuffle=True*)

Generates a random sample from a given array

#### **Parameters**

- **a**
[{array_like, int}] If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated from np.arange(a).

### **size**

[{int, tuple[int]}, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn from the 1-d *a*. If *a* has more than one dimension, the *size* shape will be inserted into the *axis* dimension, so the output ndim will be a.ndim - 1 + len(size). Default is None, in which case a single value is returned.

#### **replace**

[bool, optional] Whether the sample is with or without replacement. Default is True, meaning that a value of a can be selected multiple times.

### **p**

[1-D array_like, optional] The probabilities associated with each entry in a. If not given, the sample assumes a uniform distribution over all entries in a.

#### **axis**

[int, optional] The axis along which the selection is performed. The default, 0, selects by row.

#### **shuffle**

[bool, optional] Whether the sample is shuffled when sampling without replacement. Default is True, False provides a speedup.

### **Returns**

#### **samples**

[single item or ndarray] The generated random samples

#### **Raises**

#### **ValueError**

If a is an int and less than zero, if p is not 1-dimensional, if a is array-like with a size 0, if p is not a vector of probabilities, if a and p have different lengths, or if replace=False and the sample size is greater than the population size.

#### **See also:**

*integers***,** *shuffle***,** *permutation*

#### **Notes**

Setting user-specified probabilities through p uses a more general but less efficient sampler than the default. The general sampler produces a different sample than the optimized sampler even if each element of p is 1 / len(a).

p must sum to 1 when cast to float64. To ensure this, you may wish to normalize using p = p / np.sum(p, dtype=float).

When passing a as an integer type and size is not specified, the return type is a native Python int.

#### **Examples**

Generate a uniform random sample from np.arange(5) of size 3:

```
>>> rng = np.random.default_rng()
>>> rng.choice(5, 3)
array([0, 3, 4]) # random
>>> #This is equivalent to rng.integers(0,5,3)
```
Generate a non-uniform random sample from np.arange(5) of size 3:

```
>>> rng.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
array([3, 3, 0]) # random
```
Generate a uniform random sample from np.arange(5) of size 3 without replacement:

```
>>> rng.choice(5, 3, replace=False)
array([3,1,0]) # random
>>> #This is equivalent to rng.permutation(np.arange(5))[:3]
```
Generate a uniform random sample from a 2-D array along the first axis (the default), without replacement:

```
>>> rng.choice([[0, 1, 2], [3, 4, 5], [6, 7, 8]], 2, replace=False)
array([[3, 4, 5], # random
       [0, 1, 2]])
```
Generate a non-uniform random sample from np.arange(5) of size 3 without replacement:

```
>>> rng.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
array([2, 3, 0]) # random
```
Any of the above can be repeated with an arbitrary array-like instead of just integers. For instance:

```
>>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
>>> rng.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'], # random
      dtype='<U11')
```
method

random.Generator.**bytes**(*length*)

Return random bytes.

#### **Parameters**

**length** [int] Number of random bytes.

#### **Returns**

**out**

[bytes] String of length *length*.

#### **Notes**

This function generates random bytes from a discrete uniform distribution. The generated bytes are independent from the CPU's native endianness.

### **Examples**

```
>>> rng = np.random.default_rng()
>>> rng.bytes(10)
b'\xfeC\x9b\x86\x17\xf2\xa1\xafcp' # random
```
### **Permutations**

The methods for randomly permuting a sequence are

| shuffle(x[, axis]) | Modify an array or sequence in-place by shuffling its con |
| --- | --- |
|  | tents. |
| permutation(x[, axis]) | Randomly permute a sequence, or return a permuted |
|  | range. |
| permuted(x[, axis, out]) | Randomly permute x along axis axis. |

method

```
random.Generator.shuffle(x, axis=0)
```
Modify an array or sequence in-place by shuffling its contents.

The order of sub-arrays is changed but their contents remains the same.

#### **Parameters**

**x**

[ndarray or MutableSequence] The array, list or mutable sequence to be shuffled.

**axis**

[int, optional] The axis which *x* is shuffled along. Default is 0. It is only supported on *ndarray* objects.

#### **Returns**

**None**

**See also:**

*permuted permutation*

#### **Notes**

An important distinction between methods shuffle and permuted is how they both treat the axis parameter which can be found at *Handling the axis parameter*.

#### **Examples**

```
>>> rng = np.random.default_rng()
>>> arr = np.arange(10)
>>> arr
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> rng.shuffle(arr)
>>> arr
array([2, 0, 7, 5, 1, 4, 8, 9, 3, 6]) # random
```

```
>>> arr = np.arange(9).reshape((3, 3))
>>> arr
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> rng.shuffle(arr)
>>> arr
array([[3, 4, 5], # random
       [6, 7, 8],
       [0, 1, 2]])
```

```
>>> arr = np.arange(9).reshape((3, 3))
>>> arr
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> rng.shuffle(arr, axis=1)
>>> arr
```
(continues on next page)

(continued from previous page)

```
array([[2, 0, 1], # random
       [5, 3, 4],
       [8, 6, 7]])
```
#### method

random.Generator.**permutation**(*x*, *axis=0*)

Randomly permute a sequence, or return a permuted range.

#### **Parameters**

#### **x**

[int or array_like] If *x* is an integer, randomly permute np.arange(x). If *x* is an array, make a copy and shuffle the elements randomly.

#### **axis**

[int, optional] The axis which *x* is shuffled along. Default is 0.

#### **Returns**

#### **out**

[ndarray] Permuted sequence or array range.

### **Examples**

```
>>> rng = np.random.default_rng()
>>> rng.permutation(10)
array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random
```

```
>>> rng.permutation([1, 4, 9, 12, 15])
array([15, 1, 9, 4, 12]) # random
```

```
>>> arr = np.arange(9).reshape((3, 3))
>>> rng.permutation(arr)
array([[6, 7, 8], # random
       [0, 1, 2],
       [3, 4, 5]])
```

```
>>> rng.permutation("abc")
Traceback (most recent call last):
    ...
numpy.exceptions.AxisError: axis 0 is out of bounds for array of dimension 0
```

```
>>> arr = np.arange(9).reshape((3, 3))
>>> rng.permutation(arr, axis=1)
array([[0, 2, 1], # random
       [3, 5, 4],
       [6, 8, 7]])
```
method

random.Generator.**permuted**(*x*, *axis=None*, *out=None*)

Randomly permute *x* along axis *axis*.

Unlike *shuffle*, each slice along the given axis is shuffled independently of the others.

#### **Parameters**

# **x**

[array_like, at least one-dimensional] Array to be shuffled.

#### **axis**

[int, optional] Slices of *x* in this axis are shuffled. Each slice is shuffled independently of the others. If *axis* is None, the flattened array is shuffled.

#### **out**

[ndarray, optional] If given, this is the destination of the shuffled array. If *out* is None, a shuffled copy of the array is returned.

### **Returns**

#### **ndarray**

If *out* is None, a shuffled copy of *x* is returned. Otherwise, the shuffled array is stored in *out*, and *out* is returned

#### **See also:**

### *shuffle permutation*

#### **Notes**

An important distinction between methods shuffle and permuted is how they both treat the axis parameter which can be found at *Handling the axis parameter*.

#### **Examples**

Create a *numpy.random.Generator* instance:

```
>>> rng = np.random.default_rng()
```
Create a test array:

```
>>> x = np.arange(24).reshape(3, 8)
>>> x
array([[ 0, 1, 2, 3, 4, 5, 6, 7],
      [ 8, 9, 10, 11, 12, 13, 14, 15],
      [16, 17, 18, 19, 20, 21, 22, 23]])
```
Shuffle the rows of *x*:

```
>>> y = rng.permuted(x, axis=1)
>>> y
array([[ 4, 3, 6, 7, 1, 2, 5, 0], # random
      [15, 10, 14, 9, 12, 11, 8, 13],
      [17, 16, 20, 21, 18, 22, 23, 19]])
```
*x* has not been modified:

**>>>** x array([[ 0, 1, 2, 3, 4, 5, 6, 7], [ 8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23]]) To shuffle the rows of *x* in-place, pass *x* as the *out* parameter:

```
>>> y = rng.permuted(x, axis=1, out=x)
>>> x
array([[ 3, 0, 4, 7, 1, 6, 2, 5], # random
      [ 8, 14, 13, 9, 12, 11, 15, 10],
      [17, 18, 16, 22, 19, 23, 20, 21]])
```
Note that when the out parameter is given, the return value is out:

```
>>> y is x
True
```
The following table summarizes the behaviors of the methods.

| method | copy/in-place | axis handling |
| --- | --- | --- |
| shuffle | in-place | as if 1d |
| permutation | copy | as if 1d |
| permuted | either (use 'out' for in-place) | axis independent |

The following subsections provide more details about the differences.

### **In-place vs. copy**

The main difference between *Generator.shuffle* and *Generator.permutation* is that *Generator. shuffle* operates in-place, while *Generator.permutation* returns a copy.

By default, *Generator.permuted* returns a copy. To operate in-place with *Generator.permuted*, pass the same array as the first argument *and* as the value of the out parameter. For example,

```
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> x = np.arange(0, 15).reshape(3, 5)
>>> x
array([[ 0, 1, 2, 3, 4],
      [ 5, 6, 7, 8, 9],
      [10, 11, 12, 13, 14]])
>>> y = rng.permuted(x, axis=1, out=x)
>>> x
array([[ 1, 0, 2, 4, 3], # random
      [ 6, 7, 8, 9, 5],
      [10, 14, 11, 13, 12]])
```
Note that when out is given, the return value is out:

**>>>** y **is** x True

### **Handling the axis parameter**

An important distinction for these methods is how they handle the axis parameter. Both *Generator.shuffle* and *Generator.permutation* treat the input as a one-dimensional sequence, and the axis parameter determines which dimension of the input array to use as the sequence. In the case of a two-dimensional array, axis=0 will, in effect, rearrange the rows of the array, and axis=1 will rearrange the columns. For example

```
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> x = np.arange(0, 15).reshape(3, 5)
>>> x
array([[ 0, 1, 2, 3, 4],
      [ 5, 6, 7, 8, 9],
      [10, 11, 12, 13, 14]])
>>> rng.permutation(x, axis=1)
array([[ 1, 3, 2, 0, 4], # random
      [ 6, 8, 7, 5, 9],
      [11, 13, 12, 10, 14]])
```
Note that the columns have been rearranged "in bulk": the values within each column have not changed.

The method *Generator.permuted* treats the axis parameter similar to how *numpy.sort* treats it. Each slice along the given axis is shuffled independently of the others. Compare the following example of the use of *Generator. permuted* to the above example of *Generator.permutation*:

```
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> rng.permuted(x, axis=1)
array([[ 1, 0, 2, 4, 3], # random
      [ 5, 7, 6, 9, 8],
      [10, 14, 12, 13, 11]])
```
In this example, the values within each row (i.e. the values along axis=1) have been shuffled independently. This is not a "bulk" shuffle of the columns.

### **Shuffling non-NumPy sequences**

*Generator.shuffle* works on non-NumPy sequences. That is, if it is given a sequence that is not a NumPy array, it shuffles that sequence in-place.

```
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> a = ['A', 'B', 'C', 'D', 'E']
>>> rng.shuffle(a) # shuffle the list in-place
>>> a
['B', 'D', 'A', 'E', 'C'] # random
```
### **Distributions**

| beta(a, b[, size]) | Draw samples from a Beta distribution. |
| --- | --- |
| binomial(n, p[, size]) | Draw samples from a binomial distribution. |
| chisquare(df[, size]) | Draw samples from a chi-square distribution. |
| dirichlet(alpha[, size]) | Draw samples from the Dirichlet distribution. |
| exponential([scale, size]) | Draw samples from an exponential distribution. |
| f(dfnum, dfden[, size]) | Draw samples from an F distribution. |
| gamma(shape[, scale, size]) | Draw samples from a Gamma distribution. |
| geometric(p[, size]) | Draw samples from the geometric distribution. |
| gumbel([loc, scale, size]) | Draw samples from a Gumbel distribution. |
| hypergeometric(ngood, nbad, nsample[, size]) | Draw samples from a Hypergeometric distribution. |
| laplace([loc, scale, size]) | Draw samples from the Laplace or double exponential |
|  | distribution with specified location (or mean) and scale |
|  | (decay). |
| logistic([loc, scale, size]) | Draw samples from a logistic distribution. |
| lognormal([mean, sigma, size]) | Draw samples from a log-normal distribution. |
| logseries(p[, size]) | Draw samples from a logarithmic series distribution. |
| multinomial(n, pvals[, size]) | Draw samples from a multinomial distribution. |
| multivariate_hypergeometric(colors, nsam | Generate variates from a multivariate hypergeometric dis |
| ple) | tribution. |
| multivariate_normal(mean, cov[, size, ...]) | Draw random samples from a multivariate normal distri |
|  | bution. |
| negative_binomial(n, p[, size]) | Draw samples from a negative binomial distribution. |
| noncentral_chisquare(df, nonc[, size]) | Draw samples from a noncentral chi-square distribution. |
| noncentral_f(dfnum, dfden, nonc[, size]) | Draw samples from the noncentral F distribution. |
| normal([loc, scale, size]) | Draw random samples from a normal (Gaussian) distri |
|  | bution. |
| pareto(a[, size]) | Draw samples from a Pareto II (AKA Lomax) distribution |
|  | with specified shape. |
| poisson([lam, size]) | Draw samples from a Poisson distribution. |
| power(a[, size]) | Draws samples in [0, 1] from a power distribution with |
|  | positive exponent a - 1. |
| rayleigh([scale, size]) | Draw samples from a Rayleigh distribution. |
| standard_cauchy([size]) | Draw samples from a standard Cauchy distribution with |
|  | mode = 0. |
| standard_exponential([size, dtype, method, | Draw samples from the standard exponential distribution. |
| out]) |  |
| standard_gamma(shape[, size, dtype, out]) | Draw samples from a standard Gamma distribution. |
| standard_normal([size, dtype, out]) | Draw samples from a standard Normal distribution |
|  | (mean=0, stdev=1). |
| standard_t(df[, size]) | Draw samples from a standard Student's t distribution |
|  | with df degrees of freedom. |
| triangular(left, mode, right[, size]) | Draw samples from the triangular distribution over the in |
|  | terval [left, right]. |
| uniform([low, high, size]) | Draw samples from a uniform distribution. |
| vonmises(mu, kappa[, size]) | Draw samples from a von Mises distribution. |
| wald(mean, scale[, size]) | Draw samples from a Wald, or inverse Gaussian, distri |
|  | bution. |
| weibull(a[, size]) | Draw samples from a Weibull distribution. |
| zipf(a[, size]) | Draw samples from a Zipf distribution. |

method

random.Generator.**beta**(*a*, *b*, *size=None*)

Draw samples from a Beta distribution.

The Beta distribution is a special case of the Dirichlet distribution, and is related to the Gamma distribution. It has the probability distribution function

$$f(x;a,b)=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1},$$

where the normalization, B, is the beta function,

$$B(\alpha,\beta)=\int_{0}^{1}t^{\alpha-1}(1-t)^{\beta-1}d t.$$

It is often seen in Bayesian inference and order statistics.

**Parameters**

**a**

[float or array_like of floats] Alpha, positive (>0).

**b**

[float or array_like of floats] Beta, positive (>0).

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if a and b are both scalars. Otherwise, np.broadcast(a, b).size samples are drawn.

# **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized beta distribution.

#### **References**

### [1]

#### **Examples**

The beta distribution has mean a/(a+b). If a == b and both are > 1, the distribution is symmetric with mean 0.5.

```
>>> rng = np.random.default_rng()
>>> a, b, size = 2.0, 2.0, 10000
>>> sample = rng.beta(a=a, b=b, size=size)
>>> np.mean(sample)
0.5047328775385895 # may vary
```
Otherwise the distribution is skewed left or right according to whether a or b is greater. The distribution is mirror symmetric. See for example:

```
>>> a, b, size = 2, 7, 10000
>>> sample_left = rng.beta(a=a, b=b, size=size)
>>> sample_right = rng.beta(a=b, b=a, size=size)
>>> m_left, m_right = np.mean(sample_left), np.mean(sample_right)
>>> print(m_left, m_right)
```
(continues on next page)

(continued from previous page)

```
0.2238596793678923 0.7774613834041182 # may vary
>>> print(m_left - a/(a+b))
0.001637457145670096 # may vary
>>> print(m_right - b/(a+b))
-0.0003163943736596009 # may vary
```
Display the histogram of the two samples:

```
>>> import matplotlib.pyplot as plt
>>> plt.hist([sample_left, sample_right],
... 50, density=True, histtype='bar')
>>> plt.show()
```
![](_page_131_Figure_5.jpeg)

#### method

random.Generator.**binomial**(*n*, *p*, *size=None*)

Draw samples from a binomial distribution.

Samples are drawn from a binomial distribution with specified parameters, n trials and p probability of success where n an integer >= 0 and p is in the interval [0,1]. (n may be input as a float, but it is truncated to an integer in use)

### **Parameters**

**n**

[int or array_like of ints] Parameter of the distribution, >= 0. Floats are also accepted, but they will be truncated to integers.

**p**

[float or array_like of floats] Parameter of the distribution, >= 0 and <=1.

### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if n and p are both scalars. Otherwise, np.broadcast(n, p).size samples are drawn.

#### **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized binomial distribution, where each sample is equal to the number of successes over the n trials.

**See also:**

```
scipy.stats.binom
```
probability density function, distribution or cumulative density function, etc.

#### **Notes**

The probability mass function (PMF) for the binomial distribution is

$$P(N)={\binom{n}{N}}p^{N}(1-p)^{n-N},$$

where *n* is the number of trials, *p* is the probability of success, and *N* is the number of successes.

When estimating the standard error of a proportion in a population by using a random sample, the normal distribution works well unless the product p*n <=5, where p = population proportion estimate, and n = number of samples, in which case the binomial distribution is used instead. For example, a sample of 15 people shows 4 who are left handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4, so the binomial distribution should be used in this case.

#### **References**

[1], [2], [3], [4], [5]

#### **Examples**

Draw samples from the distribution:

```
>>> rng = np.random.default_rng()
>>> n, p, size = 10, .5, 10000
>>> s = rng.binomial(n, p, 10000)
```
Assume a company drills 9 wild-cat oil exploration wells, each with an estimated probability of success of p=0.1. All nine wells fail. What is the probability of that happening?

Over size = 20,000 trials the probability of this happening is on average:

```
>>> n, p, size = 9, 0.1, 20000
>>> np.sum(rng.binomial(n=n, p=p, size=size) == 0)/size
0.39015 # may vary
```
The following can be used to visualize a sample with n=100, p=0.4 and the corresponding probability density function:

```
>>> import matplotlib.pyplot as plt
>>> from scipy.stats import binom
>>> n, p, size = 100, 0.4, 10000
>>> sample = rng.binomial(n, p, size=size)
>>> count, bins, _ = plt.hist(sample, 30, density=True)
>>> x = np.arange(n)
>>> y = binom.pmf(x, n, p)
>>> plt.plot(x, y, linewidth=2, color='r')
```
![](_page_133_Figure_1.jpeg)

### method

random.Generator.**chisquare**(*df*, *size=None*)

Draw samples from a chi-square distribution.

When *df* independent random variables, each with standard normal distributions (mean 0, variance 1), are squared and summed, the resulting distribution is chi-square (see Notes). This distribution is often used in hypothesis testing.

### **Parameters**

### **df**

[float or array_like of floats] Number of degrees of freedom, must be > 0.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if df is a scalar. Otherwise, np.array(df).size samples are drawn.

### **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized chi-square distribution.

### **Raises**

### **ValueError**

When *df* <= 0 or when an inappropriate *size* (e.g. size=-1) is given.

#### **Notes**

The variable obtained by summing the squares of *df* independent, standard normally distributed random variables:

$$Q=\sum_{i=1}^{\mathbb{d}\mathbb{f}}X_{i}^{2}$$

is chi-square distributed, denoted

$$Q\sim\chi_{k}^{2}.$$

The probability density function of the chi-squared distribution is

$$p(x)={\frac{(1/2)^{k/2}}{\Gamma(k/2)}}x^{k/2-1}e^{-x/2},$$

where Γ is the gamma function,

$$\Gamma(x)=\int_{0}^{-\infty}t^{x-1}e^{-t}d t.$$

#### **References**

[1]

#### **Examples**

**>>>** rng = np.random.default_rng() **>>>** rng.chisquare(2,4) array([ 1.89920014, 9.00867716, 3.13710533, 5.62318272]) # random

The distribution of a chi-square random variable with 20 degrees of freedom looks as follows:

```
>>> import matplotlib.pyplot as plt
>>> import scipy.stats as stats
>>> s = rng.chisquare(20, 10000)
>>> count, bins, _ = plt.hist(s, 30, density=True)
>>> x = np.linspace(0, 60, 1000)
>>> plt.plot(x, stats.chi2.pdf(x, df=20))
>>> plt.xlim([0, 60])
>>> plt.show()
```
#### method

random.Generator.**dirichlet**(*alpha*, *size=None*)

Draw samples from the Dirichlet distribution.

Draw *size* samples of dimension k from a Dirichlet distribution. A Dirichlet-distributed random variable can be seen as a multivariate generalization of a Beta distribution. The Dirichlet distribution is a conjugate prior of a multinomial distribution in Bayesian inference.

#### **Parameters**

**alpha**

[sequence of floats, length k] Parameter of the distribution (length k for sample of length k).

![](_page_135_Figure_1.jpeg)

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n), then m * n

* k samples are drawn. Default is None, in which case a vector of length k is returned.

### **Returns**

#### **samples**

[ndarray,] The drawn samples, of shape (size, k).

#### **Raises**

**ValueError**

If any value in alpha is less than zero

#### **Notes**

The Dirichlet distribution is a distribution over vectors *x* that fulfil the conditions *xi >* 0 and P*k i*=1 *xi* = 1. The probability density function *p* of a Dirichlet-distributed random vector *X* is proportional to

$$p(x)\propto\prod_{i=1}^{k}x_{i}^{\alpha_{i}-1},$$

where *α* is a vector containing the positive concentration parameters.

The method uses the following property for computation: let *Y* be a random vector which has components that follow a standard gamma distribution, then *X* = ∑ 1 *k i*=1 *Yi Y* is Dirichlet-distributed

### **References**

[1], [2]

### **Examples**

Taking an example cited in Wikipedia, this distribution can be used if one wanted to cut strings (each of initial length 1.0) into K pieces with different lengths, where each piece had, on average, a designated average length, but allowing some variation in the relative sizes of the pieces.

```
>>> rng = np.random.default_rng()
>>> s = rng.dirichlet((10, 5, 3), 20).transpose()
>>> import matplotlib.pyplot as plt
>>> plt.barh(range(20), s[0])
>>> plt.barh(range(20), s[1], left=s[0], color='g')
>>> plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
>>> plt.title("Lengths of Strings")
```
![](_page_136_Figure_6.jpeg)

#### method

random.Generator.**exponential**(*scale=1.0*, *size=None*)

Draw samples from an exponential distribution.

Its probability density function is

$$f(x;{\frac{1}{\beta}})={\frac{1}{\beta}}\exp(-{\frac{x}{\beta}}),$$

for x > 0 and 0 elsewhere. *β* is the scale parameter, which is the inverse of the rate parameter *λ* = 1/*β*. The rate parameter is an alternative, widely used parameterization of the exponential distribution [3].

The exponential distribution is a continuous analogue of the geometric distribution. It describes many common situations, such as the size of raindrops measured over many rainstorms [1], or the time between page requests to Wikipedia [2].

#### **Parameters**

#### **scale**

[float or array_like of floats] The scale parameter, *β* = 1/*λ*. Must be non-negative.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if scale is a scalar. Otherwise, np.array(scale).size samples are drawn.

### **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized exponential distribution.

#### **References**

### [1], [2], [3]

#### **Examples**

Assume a company has 10000 customer support agents and the time between customer calls is exponentially distributed and that the average time between customer calls is 4 minutes.

```
>>> scale, size = 4, 10000
>>> rng = np.random.default_rng()
>>> time_between_calls = rng.exponential(scale=scale, size=size)
```
What is the probability that a customer will call in the next 4 to 5 minutes?

```
>>> x = ((time_between_calls < 5).sum())/size
>>> y = ((time_between_calls < 4).sum())/size
>>> x - y
0.08 # may vary
```
The corresponding distribution can be visualized as follows:

```
>>> import matplotlib.pyplot as plt
>>> scale, size = 4, 10000
>>> rng = np.random.default_rng()
>>> sample = rng.exponential(scale=scale, size=size)
>>> count, bins, _ = plt.hist(sample, 30, density=True)
>>> plt.plot(bins, scale**(-1)*np.exp(-scale**-1*bins), linewidth=2, color='r')
>>> plt.show()
```
#### method

random.Generator.**f**(*dfnum*, *dfden*, *size=None*)

Draw samples from an F distribution.

Samples are drawn from an F distribution with specified parameters, *dfnum* (degrees of freedom in numerator) and *dfden* (degrees of freedom in denominator), where both parameters must be greater than zero.

The random variate of the F distribution (also known as the Fisher distribution) is a continuous probability distribution that arises in ANOVA tests, and is the ratio of two chi-square variates.

#### **Parameters**

#### **dfnum**

[float or array_like of floats] Degrees of freedom in numerator, must be > 0.

![](_page_138_Figure_1.jpeg)

#### **dfden**

[float or array_like of float] Degrees of freedom in denominator, must be > 0.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if dfnum and dfden are both scalars. Otherwise, np.broadcast(dfnum, dfden).size samples are drawn.

#### **Returns**

#### **out**

[ndarray or scalar] Drawn samples from the parameterized Fisher distribution.

### **See also:**

#### **scipy.stats.f**

probability density function, distribution or cumulative density function, etc.

#### **Notes**

The F statistic is used to compare in-group variances to between-group variances. Calculating the distribution depends on the sampling, and so it is a function of the respective degrees of freedom in the problem. The variable *dfnum* is the number of samples minus one, the between-groups degrees of freedom, while *dfden* is the withingroups degrees of freedom, the sum of the number of samples in each group minus the number of groups.

### **References**

[1], [2]

# **Examples**

An example from Glantz[1], pp 47-40:

Two groups, children of diabetics (25 people) and children from people without diabetes (25 controls). Fasting blood glucose was measured, case group had a mean value of 86.1, controls had a mean value of 82.2. Standard deviations were 2.09 and 2.49 respectively. Are these data consistent with the null hypothesis that the parents diabetic status does not affect their children's blood glucose levels? Calculating the F statistic from the data gives a value of 36.01.

Draw samples from the distribution:

```
>>> dfnum = 1. # between group degrees of freedom
>>> dfden = 48. # within groups degrees of freedom
>>> rng = np.random.default_rng()
>>> s = rng.f(dfnum, dfden, 1000)
```
The lower bound for the top 1% of the samples is :

```
>>> np.sort(s)[-10]
7.61988120985 # random
```
So there is about a 1% chance that the F statistic will exceed 7.62, the measured value is 36, so the null hypothesis is rejected at the 1% level.

The corresponding probability density function for n = 20 and m = 20 is:

```
>>> import matplotlib.pyplot as plt
>>> from scipy import stats
>>> dfnum, dfden, size = 20, 20, 10000
>>> s = rng.f(dfnum=dfnum, dfden=dfden, size=size)
>>> bins, density, _ = plt.hist(s, 30, density=True)
>>> x = np.linspace(0, 5, 1000)
>>> plt.plot(x, stats.f.pdf(x, dfnum, dfden))
>>> plt.xlim([0, 5])
>>> plt.show()
```
method

random.Generator.**gamma**(*shape*, *scale=1.0*, *size=None*)

Draw samples from a Gamma distribution.

Samples are drawn from a Gamma distribution with specified parameters, *shape* (sometimes designated "k") and *scale* (sometimes designated "theta"), where both parameters are > 0.

### **Parameters**

#### **shape**

[float or array_like of floats] The shape of the gamma distribution. Must be non-negative.

#### **scale**

[float or array_like of floats, optional] The scale of the gamma distribution. Must be nonnegative. Default is equal to 1.

![](_page_140_Figure_1.jpeg)

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if shape and scale are both scalars. Otherwise, np.broadcast(shape, scale).size samples are drawn.

### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized gamma distribution.

#### **See also:**

#### **scipy.stats.gamma**

probability density function, distribution or cumulative density function, etc.

#### **Notes**

The probability density for the Gamma distribution is

$$p(x)=x^{k-1}\frac{e^{-x/\theta}}{\theta^{k}\Gamma(k)},$$

where *k* is the shape and *θ* the scale, and Γ is the Gamma function.

The Gamma distribution is often used to model the times to failure of electronic components, and arises naturally in processes for which the waiting times between Poisson distributed events are relevant.

### **References**

[1], [2]

### **Examples**

Draw samples from the distribution:

```
>>> shape, scale = 2., 2. # mean=4, std=2*sqrt(2)
>>> rng = np.random.default_rng()
>>> s = rng.gamma(shape, scale, 1000)
```
Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> import scipy.special as sps
>>> count, bins, _ = plt.hist(s, 50, density=True)
>>> y = bins**(shape-1)*(np.exp(-bins/scale) /
... (sps.gamma(shape)*scale**shape))
>>> plt.plot(bins, y, linewidth=2, color='r')
>>> plt.show()
```
![](_page_141_Figure_8.jpeg)

method

random.Generator.**geometric**(*p*, *size=None*)

Draw samples from the geometric distribution.

Bernoulli trials are experiments with one of two outcomes: success or failure (an example of such an experiment is flipping a coin). The geometric distribution models the number of trials that must be run in order to achieve success. It is therefore supported on the positive integers, k = 1, 2, ....

The probability mass function of the geometric distribution is

$${f{{\left({k}\right)}}}={\left({1}-{p}\right)}^{{{k}-{1}}}{p}$$

where *p* is the probability of success of an individual trial.

#### **Parameters**

**p**

[float or array_like of floats] The probability of success of an individual trial.

**size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if p is a scalar. Otherwise, np.array(p).size samples are drawn.

### **Returns**

#### **out**

[ndarray or scalar] Drawn samples from the parameterized geometric distribution.

#### **References**

[1]

### **Examples**

Draw 10,000 values from the geometric distribution, with the probability of an individual success equal to p = 0.35:

```
>>> p, size = 0.35, 10000
>>> rng = np.random.default_rng()
>>> sample = rng.geometric(p=p, size=size)
```
What proportion of trials succeeded after a single run?

```
>>> (sample == 1).sum()/size
0.34889999999999999 # may vary
```
The geometric distribution with p=0.35 looks as follows:

```
>>> import matplotlib.pyplot as plt
>>> count, bins, _ = plt.hist(sample, bins=30, density=True)
>>> plt.plot(bins, (1-p)**(bins-1)*p)
>>> plt.xlim([0, 25])
>>> plt.show()
```
method

random.Generator.**gumbel**(*loc=0.0*, *scale=1.0*, *size=None*)

Draw samples from a Gumbel distribution.

Draw samples from a Gumbel distribution with specified location and scale. For more information on the Gumbel distribution, see Notes and References below.

#### **Parameters**

### **loc**

[float or array_like of floats, optional] The location of the mode of the distribution. Default is 0.

#### **scale**

[float or array_like of floats, optional] The scale parameter of the distribution. Default is 1. Must be non- negative.

![](_page_143_Figure_1.jpeg)

### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.

#### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized Gumbel distribution.

#### **See also:**

**scipy.stats.gumbel_l scipy.stats.gumbel_r scipy.stats.genextreme** *weibull*

#### **Notes**

The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme Value Type I) distribution is one of a class of Generalized Extreme Value (GEV) distributions used in modeling extreme value problems. The Gumbel is a special case of the Extreme Value Type I distribution for maximums from distributions with "exponential-like" tails.

The probability density for the Gumbel distribution is

$$p(x)={\frac{e^{-(x-\mu)/\beta}}{\beta}}e^{-e^{-(x-\mu)/\beta}},$$

where *µ* is the mode, a location parameter, and *β* is the scale parameter.

The Gumbel (named for German mathematician Emil Julius Gumbel) was used very early in the hydrology literature, for modeling the occurrence of flood events. It is also used for modeling maximum wind speed and rainfall rates. It is a "fat-tailed" distribution - the probability of an event in the tail of the distribution is larger than if one used a Gaussian, hence the surprisingly frequent occurrence of 100-year floods. Floods were initially modeled as a Gaussian process, which underestimated the frequency of extreme events.

It is one of a class of extreme value distributions, the Generalized Extreme Value (GEV) distributions, which also includes the Weibull and Frechet.

The function has a mean of *µ* + 0*.*57721*β* and a variance of *π* 2 6 *β* 2 .

#### **References**

[1], [2]

#### **Examples**

Draw samples from the distribution:

```
>>> rng = np.random.default_rng()
>>> mu, beta = 0, 0.1 # location and scale
>>> s = rng.gumbel(mu, beta, 1000)
```
Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> count, bins, _ = plt.hist(s, 30, density=True)
>>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
... * np.exp( -np.exp( -(bins - mu) /beta) ),
... linewidth=2, color='r')
>>> plt.show()
```
![](_page_144_Figure_10.jpeg)

Show how an extreme value distribution can arise from a Gaussian process and compare to a Gaussian:

```
>>> means = []
>>> maxima = []
>>> for i in range(0,1000) :
... a = rng.normal(mu, beta, 1000)
... means.append(a.mean())
... maxima.append(a.max())
>>> count, bins, _ = plt.hist(maxima, 30, density=True)
```
(continues on next page)

(continued from previous page)

```
>>> beta = np.std(maxima) * np.sqrt(6) / np.pi
>>> mu = np.mean(maxima) - 0.57721*beta
>>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
... * np.exp(-np.exp(-(bins - mu)/beta)),
... linewidth=2, color='r')
>>> plt.plot(bins, 1/(beta * np.sqrt(2 * np.pi))
... * np.exp(-(bins - mu)**2 / (2 * beta**2)),
... linewidth=2, color='g')
>>> plt.show()
```
![](_page_145_Figure_3.jpeg)

#### method

random.Generator.**hypergeometric**(*ngood*, *nbad*, *nsample*, *size=None*)

Draw samples from a Hypergeometric distribution.

Samples are drawn from a hypergeometric distribution with specified parameters, *ngood* (ways to make a good selection), *nbad* (ways to make a bad selection), and *nsample* (number of items sampled, which is less than or equal to the sum ngood + nbad).

#### **Parameters**

#### **ngood**

[int or array_like of ints] Number of ways to make a good selection. Must be nonnegative and less than 10**9.

#### **nbad**

[int or array_like of ints] Number of ways to make a bad selection. Must be nonnegative and less than 10**9.

#### **nsample**

[int or array_like of ints] Number of items sampled. Must be nonnegative and less than ngood + nbad.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if *ngood*, *nbad*, and *nsample* are all scalars. Otherwise, np.broadcast(ngood, nbad, nsample).size samples are drawn.

#### **Returns**

#### **out**

[ndarray or scalar] Drawn samples from the parameterized hypergeometric distribution. Each sample is the number of good items within a randomly selected subset of size *nsample* taken from a set of *ngood* good items and *nbad* bad items.

#### **See also:**

#### *multivariate_hypergeometric*

Draw samples from the multivariate hypergeometric distribution.

### **scipy.stats.hypergeom**

probability density function, distribution or cumulative density function, etc.

#### **Notes**

The probability mass function (PMF) for the Hypergeometric distribution is

$$P(x)={\frac{{\binom{g}{x}}{\binom{b}{n-x}}}{{\binom{g+b}{n}}}},$$

where 0 *≤ x ≤ n* and *n − b ≤ x ≤ g*

for P(x) the probability of x good results in the drawn sample, g = *ngood*, b = *nbad*, and n = *nsample*.

Consider an urn with black and white marbles in it, *ngood* of them are black and *nbad* are white. If you draw *nsample* balls without replacement, then the hypergeometric distribution describes the distribution of black balls in the drawn sample.

Note that this distribution is very similar to the binomial distribution, except that in this case, samples are drawn without replacement, whereas in the Binomial case samples are drawn with replacement (or the sample space is infinite). As the sample space becomes large, this distribution approaches the binomial.

The arguments *ngood* and *nbad* each must be less than *10**9*. For extremely large arguments, the algorithm that is used to compute the samples [4] breaks down because of loss of precision in floating point calculations. For such large values, if *nsample* is not also large, the distribution can be approximated with the binomial distribution, *binomial(n=nsample, p=ngood/(ngood + nbad))*.

#### **References**

[1], [2], [3], [4]

#### **Examples**

Draw samples from the distribution:

```
>>> rng = np.random.default_rng()
>>> ngood, nbad, nsamp = 100, 2, 10
# number of good, number of bad, and number of samples
>>> s = rng.hypergeometric(ngood, nbad, nsamp, 1000)
>>> from matplotlib.pyplot import hist
>>> hist(s)
# note that it is very unlikely to grab both bad items
```
Suppose you have an urn with 15 white and 15 black marbles. If you pull 15 marbles at random, how likely is it that 12 or more of them are one color?

```
>>> s = rng.hypergeometric(15, 15, 15, 100000)
>>> sum(s>=12)/100000. + sum(s<=3)/100000.
# answer = 0.003 ... pretty unlikely!
```
method

random.Generator.**laplace**(*loc=0.0*, *scale=1.0*, *size=None*)

Draw samples from the Laplace or double exponential distribution with specified location (or mean) and scale (decay).

The Laplace distribution is similar to the Gaussian/normal distribution, but is sharper at the peak and has fatter tails. It represents the difference between two independent, identically distributed exponential random variables.

### **Parameters**

### **loc**

[float or array_like of floats, optional] The position, *µ*, of the distribution peak. Default is 0.

### **scale**

[float or array_like of floats, optional] *λ*, the exponential decay. Default is 1. Must be nonnegative.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.

### **Returns**

# **out**

[ndarray or scalar] Drawn samples from the parameterized Laplace distribution.

### **Notes**

It has the probability density function

$$f(x;\mu,\lambda)=\frac{1}{2\lambda}\exp\left(-\frac{|x-\mu|}{\lambda}\right).$$

The first law of Laplace, from 1774, states that the frequency of an error can be expressed as an exponential function of the absolute magnitude of the error, which leads to the Laplace distribution. For many problems in economics and health sciences, this distribution seems to model the data better than the standard Gaussian distribution.

### **References**

[1], [2], [3], [4]

### **Examples**

Draw samples from the distribution

**>>>** loc, scale = 0., 1. **>>>** rng = np.random.default_rng() **>>>** s = rng.laplace(loc, scale, 1000)

Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> count, bins, _ = plt.hist(s, 30, density=True)
>>> x = np.arange(-8., 8., .01)
>>> pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
>>> plt.plot(x, pdf)
```
Plot Gaussian for comparison:

```
>>> g = (1/(scale * np.sqrt(2 * np.pi)) *
... np.exp(-(x - loc)**2 / (2 * scale**2)))
>>> plt.plot(x,g)
```
![](_page_148_Figure_8.jpeg)

#### method

random.Generator.**logistic**(*loc=0.0*, *scale=1.0*, *size=None*)

Draw samples from a logistic distribution.

Samples are drawn from a logistic distribution with specified parameters, loc (location or mean, also median), and scale (>0).

#### **Parameters**

#### **loc**

[float or array_like of floats, optional] Parameter of the distribution. Default is 0.

#### **scale**

[float or array_like of floats, optional] Parameter of the distribution. Must be non-negative. Default is 1.

### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.

#### **Returns**

# **out**

[ndarray or scalar] Drawn samples from the parameterized logistic distribution.

### **See also:**

### **scipy.stats.logistic**

probability density function, distribution or cumulative density function, etc.

### **Notes**

The probability density for the Logistic distribution is

$$P(x)={\frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^{2}}},$$

where *µ* = location and *s* = scale.

The Logistic distribution is used in Extreme Value problems where it can act as a mixture of Gumbel distributions, in Epidemiology, and by the World Chess Federation (FIDE) where it is used in the Elo ranking system, assuming the performance of each player is a logistically distributed random variable.

### **References**

### [1], [2], [3]

### **Examples**

Draw samples from the distribution:

```
>>> loc, scale = 10, 1
>>> rng = np.random.default_rng()
>>> s = rng.logistic(loc, scale, 10000)
>>> import matplotlib.pyplot as plt
>>> count, bins, _ = plt.hist(s, bins=50, label='Sampled data')
```
# plot sampled data against the exact distribution

```
>>> def logistic(x, loc, scale):
... return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)
>>> logistic_values = logistic(bins, loc, scale)
>>> bin_spacing = np.mean(np.diff(bins))
>>> plt.plot(bins, logistic_values * bin_spacing * s.size, label='Logistic PDF')
>>> plt.legend()
>>> plt.show()
```
method

![](_page_150_Figure_1.jpeg)

random.Generator.**lognormal**(*mean=0.0*, *sigma=1.0*, *size=None*)

Draw samples from a log-normal distribution.

Draw samples from a log-normal distribution with specified mean, standard deviation, and array shape. Note that the mean and standard deviation are not the values for the distribution itself, but of the underlying normal distribution it is derived from.

#### **Parameters**

#### **mean**

[float or array_like of floats, optional] Mean value of the underlying normal distribution. Default is 0.

#### **sigma**

[float or array_like of floats, optional] Standard deviation of the underlying normal distribution. Must be non-negative. Default is 1.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if mean and sigma are both scalars. Otherwise, np.broadcast(mean, sigma).size samples are drawn.

#### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized log-normal distribution.

#### **See also:**

#### **scipy.stats.lognorm**

probability density function, distribution, cumulative density function, etc.

#### **Notes**

A variable *x* has a log-normal distribution if *log(x)* is normally distributed. The probability density function for the log-normal distribution is:

$$p(x)={\frac{1}{\sigma x{\sqrt{2\pi}}}}e^{(-{\frac{(\ln(x)-\mu)^{2}}{2\sigma^{2}}})}$$

where *µ* is the mean and *σ* is the standard deviation of the normally distributed logarithm of the variable. A log-normal distribution results if a random variable is the *product* of a large number of independent, identicallydistributed variables in the same way that a normal distribution results if the variable is the *sum* of a large number of independent, identically-distributed variables.

#### **References**

[1], [2]

#### **Examples**

Draw samples from the distribution:

```
>>> rng = np.random.default_rng()
>>> mu, sigma = 3., 1. # mean and standard deviation
>>> s = rng.lognormal(mu, sigma, 1000)
```
Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> count, bins, _ = plt.hist(s, 100, density=True, align='mid')
```

```
>>> x = np.linspace(min(bins), max(bins), 10000)
>>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
... / (x * sigma * np.sqrt(2 * np.pi)))
```

```
>>> plt.plot(x, pdf, linewidth=2, color='r')
>>> plt.axis('tight')
>>> plt.show()
```
**>>>** sigma = np.std(np.log(b)) **>>>** mu = np.mean(np.log(b))

Demonstrate that taking the products of random samples from a uniform distribution can be fit well by a log-normal probability density function.

```
>>> # Generate a thousand samples: each is the product of 100 random
>>> # values, drawn from a normal distribution.
>>> rng = rng
>>> b = []
>>> for i in range(1000):
... a = 10. + rng.standard_normal(100)
... b.append(np.prod(a))
>>> b = np.array(b) / np.min(b) # scale values to be positive
>>> count, bins, _ = plt.hist(b, 100, density=True, align='mid')
```
![](_page_152_Figure_1.jpeg)

```
>>> x = np.linspace(min(bins), max(bins), 10000)
>>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
... / (x * sigma * np.sqrt(2 * np.pi)))
```

```
>>> plt.plot(x, pdf, color='r', linewidth=2)
>>> plt.show()
```
![](_page_152_Figure_4.jpeg)

### method

random.Generator.**logseries**(*p*, *size=None*)

Draw samples from a logarithmic series distribution.

Samples are drawn from a log series distribution with specified shape parameter, 0 <= p < 1.

### **Parameters**

### **p**

[float or array_like of floats] Shape parameter for the distribution. Must be in the range [0, 1).

### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if p is a scalar. Otherwise, np.array(p).size samples are drawn.

### **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized logarithmic series distribution.

### **See also:**

### **scipy.stats.logser**

probability density function, distribution or cumulative density function, etc.

### **Notes**

The probability mass function for the Log Series distribution is

$$P(k)={\frac{-p^{k}}{k\ln(1-p)}},$$

where p = probability.

The log series distribution is frequently used to represent species richness and occurrence, first proposed by Fisher, Corbet, and Williams in 1943 [2]. It may also be used to model the numbers of occupants seen in cars [3].

### **References**

[1], [2], [3], [4]

### **Examples**

Draw samples from the distribution:

```
>>> a = .6
>>> rng = np.random.default_rng()
>>> s = rng.logseries(a, 10000)
>>> import matplotlib.pyplot as plt
>>> bins = np.arange(-.5, max(s) + .5 )
>>> count, bins, _ = plt.hist(s, bins=bins, label='Sample count')
```
# plot against distribution

```
>>> def logseries(k, p):
... return -p**k/(k*np.log(1-p))
>>> centres = np.arange(1, max(s) + 1)
>>> plt.plot(centres, logseries(centres, a) * s.size, 'r', label='logseries PMF')
>>> plt.legend()
>>> plt.show()
```
method

![](_page_154_Figure_1.jpeg)

random.Generator.**multinomial**(*n*, *pvals*, *size=None*)

Draw samples from a multinomial distribution.

The multinomial distribution is a multivariate generalization of the binomial distribution. Take an experiment with one of p possible outcomes. An example of such an experiment is throwing a dice, where the outcome can be 1 through 6. Each sample drawn from the distribution represents *n* such experiments. Its values, X_i = [X_0, X_1, ..., X_p], represent the number of times the outcome was i.

#### **Parameters**

#### **n**

[int or array-like of ints] Number of experiments.

#### **pvals**

[array-like of floats] Probabilities of each of the p different outcomes with shape (k0, k1, ..., kn, p). Each element pvals[i,j,...,:] must sum to 1 (however, the last element is always assumed to account for the remaining probability, as long as sum(pvals[. .., :-1], axis=-1) <= 1.0. Must have at least 1 dimension where pvals.shape[-1] > 0.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn each with p elements. Default is None where the output size is determined by the broadcast shape of n and all by the final dimension of pvals, which is denoted as b=(b0, b1, ..., bq). If size is not None, then it must be compatible with the broadcast shape b. Specifically, size must have q or more elements and size[-(q-j):] must equal bj.

#### **Returns**

#### **out**

[ndarray] The drawn samples, of shape size, if provided. When size is provided, the output shape is size + (p,) If not specified, the shape is determined by the broadcast shape of n and pvals, (b0, b1, ..., bq) augmented with the dimension of the multinomial, p, so that that output shape is (b0, b1, ..., bq, p).

Each entry out[i,j,...,:] is a p-dimensional value drawn from the distribution.

### **Examples**

Throw a dice 20 times:

```
>>> rng = np.random.default_rng()
>>> rng.multinomial(20, [1/6.]*6, size=1)
array([[4, 1, 7, 5, 2, 1]]) # random
```
It landed 4 times on 1, once on 2, etc.

Now, throw the dice 20 times, and 20 times again:

```
>>> rng.multinomial(20, [1/6.]*6, size=2)
array([[3, 4, 3, 3, 4, 3],
       [2, 4, 3, 4, 0, 7]]) # random
```
For the first run, we threw 3 times 1, 4 times 2, etc. For the second, we threw 2 times 1, 4 times 2, etc.

Now, do one experiment throwing the dice 10 time, and 10 times again, and another throwing the dice 20 times, and 20 times again:

```
>>> rng.multinomial([[10], [20]], [1/6.]*6, size=(2, 2))
array([[[2, 4, 0, 1, 2, 1],
        [1, 3, 0, 3, 1, 2]],
       [[1, 4, 4, 4, 4, 3],
        [3, 3, 2, 5, 5, 2]]]) # random
```
The first array shows the outcomes of throwing the dice 10 times, and the second shows the outcomes from throwing the dice 20 times.

A loaded die is more likely to land on number 6:

```
>>> rng.multinomial(100, [1/7.]*5 + [2/7.])
array([11, 16, 14, 17, 16, 26]) # random
```
Simulate 10 throws of a 4-sided die and 20 throws of a 6-sided die

```
>>> rng.multinomial([10, 20],[[1/4]*4 + [0]*2, [1/6]*6])
array([[2, 1, 4, 3, 0, 0],
       [3, 3, 3, 6, 1, 4]], dtype=int64) # random
```
Generate categorical random variates from two categories where the first has 3 outcomes and the second has 2.

```
>>> rng.multinomial(1, [[.1, .5, .4 ], [.3, .7, .0]])
array([[0, 0, 1],
       [0, 1, 0]], dtype=int64) # random
```
argmax(axis=-1) is then used to return the categories.

```
>>> pvals = [[.1, .5, .4 ], [.3, .7, .0]]
>>> rvs = rng.multinomial(1, pvals, size=(4,2))
>>> rvs.argmax(axis=-1)
array([[0, 1],
       [2, 0],
       [2, 1],
       [2, 0]], dtype=int64) # random
```
The same output dimension can be produced using broadcasting.

```
>>> rvs = rng.multinomial([[1]] * 4, pvals)
>>> rvs.argmax(axis=-1)
array([[0, 1],
       [2, 0],
       [2, 1],
       [2, 0]], dtype=int64) # random
```
The probability inputs should be normalized. As an implementation detail, the value of the last entry is ignored and assumed to take up any leftover probability mass, but this should not be relied on. A biased coin which has twice as much weight on one side as on the other should be sampled like so:

```
>>> rng.multinomial(100, [1.0 / 3, 2.0 / 3]) # RIGHT
array([38, 62]) # random
```
not like:

```
>>> rng.multinomial(100, [1.0, 2.0]) # WRONG
Traceback (most recent call last):
ValueError: pvals < 0, pvals > 1 or pvals contains NaNs
```
#### method

random.Generator.**multivariate_hypergeometric**(*colors*, *nsample*, *size=None*, *method='marginals'*)

Generate variates from a multivariate hypergeometric distribution.

The multivariate hypergeometric distribution is a generalization of the hypergeometric distribution.

Choose nsample items at random without replacement from a collection with N distinct types. N is the length of colors, and the values in colors are the number of occurrences of that type in the collection. The total number of items in the collection is sum(colors). Each random variate generated by this function is a vector of length N holding the counts of the different types that occurred in the nsample items.

The name colors comes from a common description of the distribution: it is the probability distribution of the number of marbles of each color selected without replacement from an urn containing marbles of different colors; colors[i] is the number of marbles in the urn with color i.

#### **Parameters**

#### **colors**

[sequence of integers] The number of each type of item in the collection from which a sample is drawn. The values in colors must be nonnegative. To avoid loss of precision in the algorithm, sum(colors) must be less than 10**9 when *method* is "marginals".

#### **nsample**

[int] The number of items selected. nsample must not be greater than sum(colors).

#### **size**

[int or tuple of ints, optional] The number of variates to generate, either an integer or a tuple holding the shape of the array of variates. If the given size is, e.g., (k, m), then k * m variates are drawn, where one variate is a vector of length len(colors), and the return value has shape (k, m, len(colors)). If *size* is an integer, the output has shape (size, len(colors)). Default is None, in which case a single variate is returned as an array with shape (len(colors),).

#### **method**

[string, optional] Specify the algorithm that is used to generate the variates. Must be 'count' or 'marginals' (the default). See the Notes for a description of the methods.

#### **Returns**

**variates**

[ndarray] Array of variates drawn from the multivariate hypergeometric distribution.

**See also:**

```
hypergeometric
```
Draw samples from the (univariate) hypergeometric distribution.

### **Notes**

The two methods do not return the same sequence of variates.

The "count" algorithm is roughly equivalent to the following numpy code:

```
choices = np.repeat(np.arange(len(colors)), colors)
selection = np.random.choice(choices, nsample, replace=False)
variate = np.bincount(selection, minlength=len(colors))
```
The "count" algorithm uses a temporary array of integers with length sum(colors).

The "marginals" algorithm generates a variate by using repeated calls to the univariate hypergeometric sampler. It is roughly equivalent to:

```
variate = np.zeros(len(colors), dtype=np.int64)
# `remaining` is the cumulative sum of `colors` from the last
# element to the first; e.g. if `colors` is [3, 1, 5], then
# `remaining` is [9, 6, 5].
remaining = np.cumsum(colors[::-1])[::-1]
for i in range(len(colors)-1):
    if nsample < 1:
        break
    variate[i] = hypergeometric(colors[i], remaining[i+1],
                               nsample)
    nsample -= variate[i]
variate[-1] = nsample
```
The default method is "marginals". For some cases (e.g. when *colors* contains relatively small integers), the "count" method can be significantly faster than the "marginals" method. If performance of the algorithm is important, test the two methods with typical inputs to decide which works best.

### **Examples**

```
>>> colors = [16, 8, 4]
>>> seed = 4861946401452
>>> gen = np.random.Generator(np.random.PCG64(seed))
>>> gen.multivariate_hypergeometric(colors, 6)
array([5, 0, 1])
>>> gen.multivariate_hypergeometric(colors, 6, size=3)
array([[5, 0, 1],
       [2, 2, 2],
       [3, 3, 0]])
>>> gen.multivariate_hypergeometric(colors, 6, size=(2, 2))
array([[[3, 2, 1],
        [3, 2, 1]],
       [[4, 1, 1],
        [3, 2, 1]]])
```
method

random.Generator.**multivariate_normal**(*mean*, *cov*, *size=None*, *check_valid='warn'*, *tol=1e-8*, ***, *method='svd'*)

Draw random samples from a multivariate normal distribution.

The multivariate normal, multinormal or Gaussian distribution is a generalization of the one-dimensional normal distribution to higher dimensions. Such a distribution is specified by its mean and covariance matrix. These parameters are analogous to the mean (average or "center") and variance (the squared standard deviation, or "width") of the one-dimensional normal distribution.

#### **Parameters**

#### **mean**

[1-D array_like, of length N] Mean of the N-dimensional distribution.

#### **cov**

[2-D array_like, of shape (N, N)] Covariance matrix of the distribution. It must be symmetric and positive-semidefinite for proper sampling.

### **size**

[int or tuple of ints, optional] Given a shape of, for example, (m,n,k), m*n*k samples are generated, and packed in an *m*-by-*n*-by-*k* arrangement. Because each sample is *N*-dimensional, the output shape is (m,n,k,N). If no shape is specified, a single (*N*-D) sample is returned.

#### **check_valid**

[{ 'warn', 'raise', 'ignore' }, optional] Behavior when the covariance matrix is not positive semidefinite.

#### **tol**

[float, optional] Tolerance when checking the singular values in covariance matrix. cov is cast to double before the check.

#### **method**

[{ 'svd', 'eigh', 'cholesky'}, optional] The cov input is used to compute a factor matrix A such that A @ A.T = cov. This argument is used to select the method used to compute the factor matrix A. The default method 'svd' is the slowest, while 'cholesky' is the fastest but less robust than the slowest method. The method *eigh* uses eigen decomposition to compute A and is faster than svd but slower than cholesky.

### **Returns**

### **out**

[ndarray] The drawn samples, of shape *size*, if that was provided. If not, the shape is (N,).

In other words, each entry out[i,j,...,:] is an N-dimensional value drawn from the distribution.

#### **Notes**

The mean is a coordinate in N-dimensional space, which represents the location where samples are most likely to be generated. This is analogous to the peak of the bell curve for the one-dimensional or univariate normal distribution.

Covariance indicates the level to which two variables vary together. From the multivariate normal distribution, we draw N-dimensional samples, *X* = [*x*1*, x*2*, ...xN* ]. The covariance matrix element *Cij* is the covariance of *xi* and *xj* . The element *Cii* is the variance of *xi* (i.e. its "spread").

Instead of specifying the full covariance matrix, popular approximations include:

- Spherical covariance (*cov* is a multiple of the identity matrix)
- Diagonal covariance (*cov* has non-negative elements, and only on the diagonal)
This geometrical property can be seen in two dimensions by plotting generated data-points:

**>>>** mean = [0, 0] **>>>** cov = [[1, 0], [0, 100]] *# diagonal covariance*

Diagonal covariance means that points are oriented along x or y-axis:

```
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> x, y = rng.multivariate_normal(mean, cov, 5000).T
>>> plt.plot(x, y, 'x')
>>> plt.axis('equal')
>>> plt.show()
```
Note that the covariance matrix must be positive semidefinite (a.k.a. nonnegative-definite). Otherwise, the behavior of this method is undefined and backwards compatibility is not guaranteed.

This function internally uses linear algebra routines, and thus results may not be identical (even up to precision) across architectures, OSes, or even builds. For example, this is likely if cov has multiple equal singular values and method is 'svd' (default). In this case, method='cholesky' may be more robust.

#### **References**

[1], [2]

#### **Examples**

```
>>> mean = (1, 2)
>>> cov = [[1, 0], [0, 1]]
>>> rng = np.random.default_rng()
>>> x = rng.multivariate_normal(mean, cov, (3, 3))
>>> x.shape
(3, 3, 2)
```
We can use a different method other than the default to factorize cov:

```
>>> y = rng.multivariate_normal(mean, cov, (3, 3), method='cholesky')
>>> y.shape
(3, 3, 2)
```
Here we generate 800 samples from the bivariate normal distribution with mean [0, 0] and covariance matrix [[6, -3], [-3, 3.5]]. The expected variances of the first and second components of the sample are 6 and 3.5, respectively, and the expected correlation coefficient is -3/sqrt(6*3.5) ≈ -0.65465.

```
>>> cov = np.array([[6, -3], [-3, 3.5]])
>>> pts = rng.multivariate_normal([0, 0], cov, size=800)
```
Check that the mean, covariance, and correlation coefficient of the sample are close to the expected values:

```
>>> pts.mean(axis=0)
array([ 0.0326911 , -0.01280782]) # may vary
>>> np.cov(pts.T)
array([[ 5.96202397, -2.85602287],
```
(continues on next page)

(continued from previous page)

```
[-2.85602287, 3.47613949]]) # may vary
>>> np.corrcoef(pts.T)[0, 1]
-0.6273591314603949 # may vary
```
We can visualize this data with a scatter plot. The orientation of the point cloud illustrates the negative correlation of the components of this sample.

```
>>> import matplotlib.pyplot as plt
>>> plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
>>> plt.axis('equal')
>>> plt.grid()
>>> plt.show()
```
![](_page_160_Figure_5.jpeg)

#### method

random.Generator.**negative_binomial**(*n*, *p*, *size=None*)

Draw samples from a negative binomial distribution.

Samples are drawn from a negative binomial distribution with specified parameters, *n* successes and *p* probability of success where *n* is > 0 and *p* is in the interval (0, 1].

#### **Parameters**

**n**

[float or array_like of floats] Parameter of the distribution, > 0.

**p**

[float or array_like of floats] Parameter of the distribution. Must satisfy 0 < p <= 1.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if n and p are both scalars. Otherwise, np.broadcast(n, p).size samples are drawn.

#### **Returns**

#### **out**

[ndarray or scalar] Drawn samples from the parameterized negative binomial distribution,

where each sample is equal to N, the number of failures that occurred before a total of n successes was reached.

#### **Notes**

The probability mass function of the negative binomial distribution is

$$P(N;n,p)=\frac{\Gamma(N+n)}{N!\Gamma(n)}p^{n}(1-p)^{N},$$

where *n* is the number of successes, *p* is the probability of success, *N* + *n* is the number of trials, and Γ is the gamma function. When *n* is an integer, Γ(*N*+*n*) *N*!Γ(*n*) = *N*+*n−*1 *N* , which is the more common form of this term in the pmf. The negative binomial distribution gives the probability of N failures given n successes, with a success on the last trial.

If one throws a die repeatedly until the third time a "1" appears, then the probability distribution of the number of non-"1"s that appear before the third "1" is a negative binomial distribution.

Because this method internally calls Generator.poisson with an intermediate random value, a ValueError is raised when the choice of *n* and *p* would result in the mean + 10 sigma of the sampled intermediate distribution exceeding the max acceptable value of the Generator.poisson method. This happens when *p* is too low (a lot of failures happen for every success) and *n* is too big ( a lot of successes are allowed). Therefore, the *n* and *p* values must satisfy the constraint:

$$n{\frac{1-p}{p}}+10n{\sqrt{n}}{\frac{1-p}{p}}<2^{63}-1-10{\sqrt{2^{63}-1}},$$

Where the left side of the equation is the derived mean + 10 sigma of a sample from the gamma distribution internally used as the *lam* parameter of a poisson sample, and the right side of the equation is the constraint for maximum value of *lam* in Generator.poisson.

#### **References**

### [1], [2]

#### **Examples**

Draw samples from the distribution:

A real world example. A company drills wild-cat oil exploration wells, each with an estimated probability of success of 0.1. What is the probability of having one success for each successive well, that is what is the probability of a single success after drilling 5 wells, after 6 wells, etc.?

```
>>> rng = np.random.default_rng()
>>> s = rng.negative_binomial(1, 0.1, 100000)
>>> for i in range(1, 11):
... probability = sum(s<i) / 100000.
... print(i, "wells drilled, probability of one success =", probability)
```
method

random.Generator.**noncentral_chisquare**(*df*, *nonc*, *size=None*)

Draw samples from a noncentral chi-square distribution.

The noncentral *χ* 2 distribution is a generalization of the *χ* 2 distribution.

#### **Parameters**

**df**

[float or array_like of floats] Degrees of freedom, must be > 0.

**nonc**

[float or array_like of floats] Non-centrality, must be non-negative.

#### **size**

```
[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then
m * n * k samples are drawn. If size is None (default), a single value is returned if df
and nonc are both scalars. Otherwise, np.broadcast(df, nonc).size samples are
drawn.
```
#### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized noncentral chi-square distribution.

### **Notes**

The probability density function for the noncentral Chi-square distribution is

$$P(x;d f,n o n c)=\sum_{i=0}^{\infty}\frac{e^{-n o n c/2}(n o n c/2)^{i}}{i!}P_{Y_{d f+2i}}(x),$$

where *Yq* is the Chi-square with q degrees of freedom.

### **References**

[1]

### **Examples**

Draw values from the distribution and plot the histogram

```
>>> rng = np.random.default_rng()
>>> import matplotlib.pyplot as plt
>>> values = plt.hist(rng.noncentral_chisquare(3, 20, 100000),
... bins=200, density=True)
>>> plt.show()
```
Draw values from a noncentral chisquare with very small noncentrality, and compare to a chisquare.

```
>>> plt.figure()
>>> values = plt.hist(rng.noncentral_chisquare(3, .0000001, 100000),
... bins=np.arange(0., 25, .1), density=True)
>>> values2 = plt.hist(rng.chisquare(3, 100000),
... bins=np.arange(0., 25, .1), density=True)
>>> plt.plot(values[1][0:-1], values[0]-values2[0], 'ob')
>>> plt.show()
```
Demonstrate how large values of non-centrality lead to a more symmetric distribution.

![](_page_163_Figure_1.jpeg)

![](_page_163_Figure_2.jpeg)

```
>>> plt.figure()
>>> values = plt.hist(rng.noncentral_chisquare(3, 20, 100000),
... bins=200, density=True)
>>> plt.show()
```
![](_page_164_Figure_2.jpeg)

#### method

```
random.Generator.noncentral_f(dfnum, dfden, nonc, size=None)
```
Draw samples from the noncentral F distribution.

Samples are drawn from an F distribution with specified parameters, *dfnum* (degrees of freedom in numerator) and *dfden* (degrees of freedom in denominator), where both parameters > 1. *nonc* is the non-centrality parameter.

#### **Parameters**

#### **dfnum**

[float or array_like of floats] Numerator degrees of freedom, must be > 0.

#### **dfden**

[float or array_like of floats] Denominator degrees of freedom, must be > 0.

#### **nonc**

[float or array_like of floats] Non-centrality parameter, the sum of the squares of the numerator means, must be >= 0.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if dfnum, dfden, and nonc are all scalars. Otherwise, np.broadcast(dfnum, dfden, nonc). size samples are drawn.

#### **Returns**

#### **out**

[ndarray or scalar] Drawn samples from the parameterized noncentral Fisher distribution.

### **Notes**

When calculating the power of an experiment (power = probability of rejecting the null hypothesis when a specific alternative is true) the non-central F statistic becomes important. When the null hypothesis is true, the F statistic follows a central F distribution. When the null hypothesis is not true, then it follows a non-central F statistic.

### **References**

[1], [2]

### **Examples**

In a study, testing for a specific alternative to the null hypothesis requires use of the Noncentral F distribution. We need to calculate the area in the tail of the distribution that exceeds the value of the F distribution for the null hypothesis. We'll plot the two probability distributions for comparison.

```
>>> rng = np.random.default_rng()
>>> dfnum = 3 # between group deg of freedom
>>> dfden = 20 # within groups degrees of freedom
>>> nonc = 3.0
>>> nc_vals = rng.noncentral_f(dfnum, dfden, nonc, 1000000)
>>> NF = np.histogram(nc_vals, bins=50, density=True)
>>> c_vals = rng.f(dfnum, dfden, 1000000)
>>> F = np.histogram(c_vals, bins=50, density=True)
>>> import matplotlib.pyplot as plt
>>> plt.plot(F[1][1:], F[0])
>>> plt.plot(NF[1][1:], NF[0])
>>> plt.show()
```
![](_page_165_Figure_8.jpeg)

### method

random.Generator.**normal**(*loc=0.0*, *scale=1.0*, *size=None*) Draw random samples from a normal (Gaussian) distribution.

The probability density function of the normal distribution, first derived by De Moivre and 200 years later by both Gauss and Laplace independently [2], is often called the bell curve because of its characteristic shape (see the example below).

The normal distributions occurs often in nature. For example, it describes the commonly occurring distribution of samples influenced by a large number of tiny, random disturbances, each with its own unique distribution [2].

### **Parameters**

**loc**

[float or array_like of floats] Mean ("centre") of the distribution.

#### **scale**

[float or array_like of floats] Standard deviation (spread or "width") of the distribution. Must be non-negative.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, scale).size samples are drawn.

### **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized normal distribution.

### **See also:**

```
scipy.stats.norm
```
probability density function, distribution or cumulative density function, etc.

### **Notes**

The probability density for the Gaussian distribution is

$$p(x)={\frac{1}{\sqrt{2\pi\sigma^{2}}}}e^{-{\frac{(x-\mu)^{2}}{2\sigma^{2}}}},$$

where *µ* is the mean and *σ* the standard deviation. The square of the standard deviation, *σ* 2 , is called the variance.

The function has its peak at the mean, and its "spread" increases with the standard deviation (the function reaches 0.607 times its maximum at *x*+*σ* and *x−σ* [2]). This implies that *normal* is more likely to return samples lying close to the mean, rather than those far away.

### **References**

[1], [2]

### **Examples**

Draw samples from the distribution:

```
>>> mu, sigma = 0, 0.1 # mean and standard deviation
>>> rng = np.random.default_rng()
>>> s = rng.normal(mu, sigma, 1000)
```
Verify the mean and the standard deviation:

```
>>> abs(mu - np.mean(s))
0.0 # may vary
```

```
>>> abs(sigma - np.std(s, ddof=1))
0.0 # may vary
```
Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> count, bins, _ = plt.hist(s, 30, density=True)
>>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
... np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
... linewidth=2, color='r')
>>> plt.show()
```
![](_page_167_Figure_9.jpeg)

Two-by-four array of samples from the normal distribution with mean 3 and standard deviation 2.5:

```
>>> rng = np.random.default_rng()
>>> rng.normal(3, 2.5, size=(2, 4))
array([[-4.49401501, 4.00950034, -1.81814867, 7.29718677], # random
      [ 0.39924804, 4.68456316, 4.99394529, 4.84057254]]) # random
```
method

random.Generator.**pareto**(*a*, *size=None*)

Draw samples from a Pareto II (AKA Lomax) distribution with specified shape.

#### **Parameters**

**a**

[float or array_like of floats] Shape of the distribution. Must be positive.

**size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if a is a scalar. Otherwise, np.array(a).size samples are drawn.

### **Returns**

**out**

[ndarray or scalar] Drawn samples from the Pareto II distribution.

### **See also:**

```
scipy.stats.pareto
    Pareto I distribution
scipy.stats.lomax
    Lomax (Pareto II) distribution
```
**scipy.stats.genpareto** Generalized Pareto distribution

#### **Notes**

The probability density for the Pareto II distribution is

$$p(x)={\frac{a}{x+1^{a+1}}},x\geq0$$

where *a >* 0 is the shape.

The Pareto II distribution is a shifted and scaled version of the Pareto I distribution, which can be found in scipy. stats.pareto.

### **References**

[1], [2], [3], [4]

#### **Examples**

Draw samples from the distribution:

```
>>> a = 3.
>>> rng = np.random.default_rng()
>>> s = rng.pareto(a, 10000)
```
Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> x = np.linspace(0, 3, 50)
>>> pdf = a / (x+1)**(a+1)
>>> plt.hist(s, bins=x, density=True, label='histogram')
>>> plt.plot(x, pdf, linewidth=2, color='r', label='pdf')
```
(continues on next page)

(continued from previous page)

```
>>> plt.xlim(x.min(), x.max())
>>> plt.legend()
>>> plt.show()
```
![](_page_169_Figure_3.jpeg)

#### method

random.Generator.**poisson**(*lam=1.0*, *size=None*)

Draw samples from a Poisson distribution.

The Poisson distribution is the limit of the binomial distribution for large N.

#### **Parameters**

### **lam**

[float or array_like of floats] Expected number of events occurring in a fixed-time interval, must be >= 0. A sequence must be broadcastable over the requested size.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if lam is a scalar. Otherwise, np.array(lam).size samples are drawn.

### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized Poisson distribution.

#### **Notes**

The probability mass function (PMF) of Poisson distribution is

$$f(k;\lambda)={\frac{\lambda^{k}e^{-\lambda}}{k!}}$$

For events with an expected separation *λ* the Poisson distribution *f*(*k*; *λ*) describes the probability of *k* events occurring within the observed interval *λ*.

Because the output is limited to the range of the C int64 type, a ValueError is raised when *lam* is within 10 sigma of the maximum representable value.

### **References**

[1], [2]

#### **Examples**

Draw samples from the distribution:

```
>>> rng = np.random.default_rng()
>>> lam, size = 5, 10000
>>> s = rng.poisson(lam=lam, size=size)
```
Verify the mean and variance, which should be approximately lam:

**>>>** s.mean(), s.var() (4.9917 5.1088311) # may vary

Display the histogram and probability mass function:

```
>>> import matplotlib.pyplot as plt
>>> from scipy import stats
>>> x = np.arange(0, 21)
>>> pmf = stats.poisson.pmf(x, mu=lam)
>>> plt.hist(s, bins=x, density=True, width=0.5)
>>> plt.stem(x, pmf, 'C1-')
>>> plt.show()
```
Draw each 100 values for lambda 100 and 500:

**>>>** s = rng.poisson(lam=(100., 500.), size=(100, 2))

method

random.Generator.**power**(*a*, *size=None*)

Draws samples in [0, 1] from a power distribution with positive exponent a - 1.

Also known as the power function distribution.

#### **Parameters**

**a**

[float or array_like of floats] Parameter of the distribution. Must be non-negative.

![](_page_171_Figure_1.jpeg)

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if a is a scalar. Otherwise, np.array(a).size samples are drawn.

### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized power distribution.

#### **Raises**

**ValueError** If a <= 0.

### **Notes**

The probability density function is

$P(x;a)=ax^{a-1},0\leq x\leq1,a>0$.  
  

The power function distribution is just the inverse of the Pareto distribution. It may also be seen as a special case of the Beta distribution.

It is used, for example, in modeling the over-reporting of insurance claims.

#### **References**

[1], [2]

### **Examples**

Draw samples from the distribution:

**>>>** rng = np.random.default_rng() **>>>** a = 5. *# shape* **>>>** samples = 1000 **>>>** s = rng.power(a, samples)

Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> count, bins, _ = plt.hist(s, bins=30)
>>> x = np.linspace(0, 1, 100)
>>> y = a*x**(a-1.)
>>> normed_y = samples*np.diff(bins)[0]*y
>>> plt.plot(x, normed_y)
>>> plt.show()
```
![](_page_172_Figure_6.jpeg)

Compare the power function distribution to the inverse of the Pareto.

```
>>> from scipy import stats
>>> rvs = rng.power(5, 1000000)
>>> rvsp = rng.pareto(5, 1000000)
>>> xx = np.linspace(0,1,100)
>>> powpdf = stats.powerlaw.pdf(xx,5)
```

```
>>> plt.figure()
>>> plt.hist(rvs, bins=50, density=True)
>>> plt.plot(xx,powpdf,'r-')
>>> plt.title('power(5)')
```

```
>>> plt.figure()
>>> plt.hist(1./(1.+rvsp), bins=50, density=True)
>>> plt.plot(xx,powpdf,'r-')
>>> plt.title('inverse of 1 + Generator.pareto(5)')
```

```
>>> plt.figure()
>>> plt.hist(1./(1.+rvsp), bins=50, density=True)
>>> plt.plot(xx,powpdf,'r-')
>>> plt.title('inverse of stats.pareto(5)')
```
![](_page_173_Figure_2.jpeg)

![](_page_173_Figure_3.jpeg)

method

random.Generator.**rayleigh**(*scale=1.0*, *size=None*)

Draw samples from a Rayleigh distribution.

The *χ* and Weibull distributions are generalizations of the Rayleigh.

### **Parameters**

### **scale**

[float or array_like of floats, optional] Scale, also equals the mode. Must be non-negative. Default is 1.

![](_page_174_Figure_1.jpeg)

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if scale is a scalar. Otherwise, np.array(scale).size samples are drawn.

### **Returns**

# **out**

[ndarray or scalar] Drawn samples from the parameterized Rayleigh distribution.

#### **Notes**

The probability density function for the Rayleigh distribution is

$$P(x;s c a l e)={\frac{x}{s c a l e^{2}}}e^{\frac{-x^{2}}{2\cdot s c a l e^{2}}}$$

The Rayleigh distribution would arise, for example, if the East and North components of the wind velocity had identical zero-mean Gaussian distributions. Then the wind speed would have a Rayleigh distribution.

#### **References**

[1], [2]

#### **Examples**

Draw values from the distribution and plot the histogram

```
>>> from matplotlib.pyplot import hist
>>> rng = np.random.default_rng()
>>> values = hist(rng.rayleigh(3, 100000), bins=200, density=True)
```
Wave heights tend to follow a Rayleigh distribution. If the mean wave height is 1 meter, what fraction of waves are likely to be larger than 3 meters?

```
>>> meanvalue = 1
>>> modevalue = np.sqrt(2 / np.pi) * meanvalue
>>> s = rng.rayleigh(modevalue, 1000000)
```
The percentage of waves larger than 3 meters is:

```
>>> 100.*sum(s>3)/1000000.
0.087300000000000003 # random
```
### method

random.Generator.**standard_cauchy**(*size=None*)

Draw samples from a standard Cauchy distribution with mode = 0.

Also known as the Lorentz distribution.

#### **Parameters**

### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

### **Returns**

### **samples**

[ndarray or scalar] The drawn samples.

### **Notes**

The probability density function for the full Cauchy distribution is

$$P(x;x_{0},\gamma)=\frac{1}{\pi\gamma\left[1+(\frac{x-x_{0}}{\gamma})^{2}\right]}$$

and the Standard Cauchy distribution just sets *x*0 = 0 and *γ* = 1

The Cauchy distribution arises in the solution to the driven harmonic oscillator problem, and also describes spectral line broadening. It also describes the distribution of values at which a line tilted at a random angle will cut the x axis.

When studying hypothesis tests that assume normality, seeing how the tests perform on data from a Cauchy distribution is a good indicator of their sensitivity to a heavy-tailed distribution, since the Cauchy looks very much like a Gaussian distribution, but with heavier tails.

### **References**

[1], [2], [3]

### **Examples**

Draw samples and plot the distribution:

```
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> s = rng.standard_cauchy(1000000)
>>> s = s[(s>-25) & (s<25)] # truncate distribution so it plots well
>>> plt.hist(s, bins=100)
>>> plt.show()
```
![](_page_176_Figure_4.jpeg)

#### method

random.Generator.**standard_exponential**(*size=None*, *dtype=np.float64*, *method='zig'*, *out=None*) Draw samples from the standard exponential distribution.

*standard_exponential* is identical to the exponential distribution with a scale parameter of 1.

#### **Parameters**

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

#### **dtype**

[dtype, optional] Desired dtype of the result, only *float64* and *float32* are supported. Byteorder must be native. The default value is np.float64.

### **method**

[str, optional] Either 'inv' or 'zig'. 'inv' uses the default inverse CDF method. 'zig' uses the much faster Ziggurat method of Marsaglia and Tsang.

#### **out**

[ndarray, optional] Alternative output array in which to place the result. If size is not None, it must have the same shape as the provided size and must match the type of the output values.

#### **Returns**

#### **out**

[float or ndarray] Drawn samples.

### **Examples**

Output a 3x8000 array:

```
>>> rng = np.random.default_rng()
>>> n = rng.standard_exponential((3, 8000))
```
### method

random.Generator.**standard_gamma**(*shape*, *size=None*, *dtype=np.float64*, *out=None*)

Draw samples from a standard Gamma distribution.

Samples are drawn from a Gamma distribution with specified parameters, shape (sometimes designated "k") and scale=1.

### **Parameters**

### **shape**

[float or array_like of floats] Parameter, must be non-negative.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if shape is a scalar. Otherwise, np.array(shape).size samples are drawn.

#### **dtype**

[dtype, optional] Desired dtype of the result, only *float64* and *float32* are supported. Byteorder must be native. The default value is np.float64.

### **out**

[ndarray, optional] Alternative output array in which to place the result. If size is not None, it must have the same shape as the provided size and must match the type of the output values.

### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized standard gamma distribution.

### **See also:**

### **scipy.stats.gamma**

probability density function, distribution or cumulative density function, etc.

### **Notes**

The probability density for the Gamma distribution is

$$p(x)=x^{k-1}\frac{e^{-x/\theta}}{\theta^{k}\Gamma(k)},$$

where *k* is the shape and *θ* the scale, and Γ is the Gamma function.

The Gamma distribution is often used to model the times to failure of electronic components, and arises naturally in processes for which the waiting times between Poisson distributed events are relevant.

### **References**

[1], [2]

### **Examples**

Draw samples from the distribution:

```
>>> shape, scale = 2., 1. # mean and width
>>> rng = np.random.default_rng()
>>> s = rng.standard_gamma(shape, 1000000)
```
Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> import scipy.special as sps
>>> count, bins, _ = plt.hist(s, 50, density=True)
>>> y = bins**(shape-1) * ((np.exp(-bins/scale))/
... (sps.gamma(shape) * scale**shape))
>>> plt.plot(bins, y, linewidth=2, color='r')
>>> plt.show()
```
![](_page_178_Figure_8.jpeg)

#### method

random.Generator.**standard_normal**(*size=None*, *dtype=np.float64*, *out=None*) Draw samples from a standard Normal distribution (mean=0, stdev=1).

#### **Parameters**

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

#### **dtype**

[dtype, optional] Desired dtype of the result, only *float64* and *float32* are supported. Byteorder must be native. The default value is np.float64.

### **out**

[ndarray, optional] Alternative output array in which to place the result. If size is not None, it must have the same shape as the provided size and must match the type of the output values.

#### **Returns**

#### **out**

[float or ndarray] A floating-point array of shape size of drawn samples, or a single sample if size was not specified.

### **See also:**

#### *normal*

Equivalent function with additional loc and scale arguments for setting the mean and standard deviation.

#### **Notes**

For random samples from the normal distribution with mean mu and standard deviation sigma, use one of:

```
mu + sigma * rng.standard_normal(size=...)
rng.normal(mu, sigma, size=...)
```
### **Examples**

```
>>> rng = np.random.default_rng()
>>> rng.standard_normal()
2.1923875335537315 # random
```

```
>>> s = rng.standard_normal(8000)
>>> s
array([ 0.6888893 , 0.78096262, -0.89086505, ..., 0.49876311, # random
      -0.38672696, -0.4685006 ]) # random
>>> s.shape
(8000,)
>>> s = rng.standard_normal(size=(3, 4, 2))
>>> s.shape
(3, 4, 2)
```
Two-by-four array of samples from the normal distribution with mean 3 and standard deviation 2.5:

```
>>> 3 + 2.5 * rng.standard_normal(size=(2, 4))
array([[-4.49401501, 4.00950034, -1.81814867, 7.29718677], # random
      [ 0.39924804, 4.68456316, 4.99394529, 4.84057254]]) # random
```
#### method

random.Generator.**standard_t**(*df*, *size=None*)

Draw samples from a standard Student's t distribution with *df* degrees of freedom.

A special case of the hyperbolic distribution. As *df* gets large, the result resembles that of the standard normal distribution (*standard_normal*).

#### **Parameters**

**df**

[float or array_like of floats] Degrees of freedom, must be > 0.

**size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if df is a scalar. Otherwise, np.array(df).size samples are drawn.

### **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized standard Student's t distribution.

#### **Notes**

The probability density function for the t distribution is

$$P(x,d f)={\frac{\Gamma({\frac{d f+1}{2}})}{\sqrt{\pi d f}\Gamma({\frac{d f}{2}})}}\Big(1+{\frac{x^{2}}{d f}}\Big)^{-(d f+1)/2}$$

The t test is based on an assumption that the data come from a Normal distribution. The t test provides a way to test whether the sample mean (that is the mean calculated from the data) is a good estimate of the true mean.

The derivation of the t-distribution was first published in 1908 by William Gosset while working for the Guinness Brewery in Dublin. Due to proprietary issues, he had to publish under a pseudonym, and so he used the name Student.

#### **References**

[1], [2]

#### **Examples**

From Dalgaard page 83 [1], suppose the daily energy intake for 11 women in kilojoules (kJ) is:

```
>>> intake = np.array([5260., 5470, 5640, 6180, 6390, 6515, 6805, 7515, \
... 7515, 8230, 8770])
```
Does their energy intake deviate systematically from the recommended value of 7725 kJ? Our null hypothesis will be the absence of deviation, and the alternate hypothesis will be the presence of an effect that could be either positive or negative, hence making our test 2-tailed.

Because we are estimating the mean and we have N=11 values in our sample, we have N-1=10 degrees of freedom. We set our significance level to 95% and compute the t statistic using the empirical mean and empirical standard deviation of our intake. We use a ddof of 1 to base the computation of our empirical standard deviation on an unbiased estimate of the variance (note: the final estimate is not unbiased due to the concave nature of the square root).

```
>>> np.mean(intake)
6753.636363636364
>>> intake.std(ddof=1)
1142.1232221373727
>>> t = (np.mean(intake)-7725)/(intake.std(ddof=1)/np.sqrt(len(intake)))
>>> t
-2.8207540608310198
```
We draw 1000000 samples from Student's t distribution with the adequate degrees of freedom.

```
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> s = rng.standard_t(10, size=1000000)
>>> h = plt.hist(s, bins=100, density=True)
```
Does our t statistic land in one of the two critical regions found at both tails of the distribution?

```
>>> np.sum(np.abs(t) < np.abs(s)) / float(len(s))
0.018318 #random < 0.05, statistic is in critical region
```
The probability value for this 2-tailed test is about 1.83%, which is lower than the 5% pre-determined significance threshold.

Therefore, the probability of observing values as extreme as our intake conditionally on the null hypothesis being true is too low, and we reject the null hypothesis of no deviation.

![](_page_181_Figure_6.jpeg)

#### method

random.Generator.**triangular**(*left*, *mode*, *right*, *size=None*)

Draw samples from the triangular distribution over the interval [left, right].

The triangular distribution is a continuous probability distribution with lower limit left, peak at mode, and upper limit right. Unlike the other distributions, these parameters directly define the shape of the pdf.

#### **Parameters**

#### **left**

[float or array_like of floats] Lower limit.

### **mode**

[float or array_like of floats] The value where the peak of the distribution occurs. The value must fulfill the condition left <= mode <= right.

#### **right**

[float or array_like of floats] Upper limit, must be larger than *left*.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m *

n * k samples are drawn. If size is None (default), a single value is returned if left, mode,

and right are all scalars. Otherwise, np.broadcast(left, mode, right).size samples are drawn.

### **Returns**

#### **out**

[ndarray or scalar] Drawn samples from the parameterized triangular distribution.

#### **Notes**

The probability density function for the triangular distribution is

$$P(x;l,m,r)={\begin{cases}{\frac{2(x-l)}{(r-l)(m-l)}}&{{\mathrm{for~}}l\leq x\leq m,}\\ {\frac{2(r-x)}{(r-l)(r-m)}}&{{\mathrm{for~}}m\leq x\leq r,}\\ 0&{{\mathrm{otherwise.}}}\end{cases}}$$

The triangular distribution is often used in ill-defined problems where the underlying distribution is not known, but some knowledge of the limits and mode exists. Often it is used in simulations.

#### **References**

### [1]

#### **Examples**

Draw values from the distribution and plot the histogram:

```
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> h = plt.hist(rng.triangular(-3, 0, 8, 100000), bins=200,
... density=True)
>>> plt.show()
```
![](_page_182_Figure_14.jpeg)

### method

random.Generator.**uniform**(*low=0.0*, *high=1.0*, *size=None*)

Draw samples from a uniform distribution.

Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high). In other words, any value within the given interval is equally likely to be drawn by *uniform*.

#### **Parameters**

- **low**
[float or array_like of floats, optional] Lower boundary of the output interval. All values generated will be greater than or equal to low. The default value is 0.

#### **high**

[float or array_like of floats] Upper boundary of the output interval. All values generated will be less than high. The high limit may be included in the returned array of floats due to floatingpoint rounding in the equation low + (high-low) * random_sample(). high - low must be non-negative. The default value is 1.0.

### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if low and high are both scalars. Otherwise, np.broadcast(low, high).size samples are drawn.

### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized uniform distribution.

#### **See also:**

### *integers*

Discrete uniform distribution, yielding integers.

#### *random*

Floats uniformly distributed over [0, 1).

### **Notes**

The probability density function of the uniform distribution is

$$p(x)={\frac{1}{b-a}}$$

anywhere within the interval [a, b), and zero elsewhere.

When high == low, values of low will be returned.

### **Examples**

Draw samples from the distribution:

```
>>> rng = np.random.default_rng()
>>> s = rng.uniform(-1,0,1000)
```
All values are within the given interval:

```
>>> np.all(s >= -1)
True
>>> np.all(s < 0)
True
```
Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> count, bins, _ = plt.hist(s, 15, density=True)
>>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
>>> plt.show()
```
![](_page_184_Figure_4.jpeg)

### method

random.Generator.**vonmises**(*mu*, *kappa*, *size=None*)

Draw samples from a von Mises distribution.

Samples are drawn from a von Mises distribution with specified mode (mu) and concentration (kappa), on the interval [-pi, pi].

The von Mises distribution (also known as the circular normal distribution) is a continuous probability distribution on the unit circle. It may be thought of as the circular analogue of the normal distribution.

#### **Parameters**

### **mu**

[float or array_like of floats] Mode ("center") of the distribution.

### **kappa**

[float or array_like of floats] Concentration of the distribution, has to be >=0.

### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if mu and kappa are both scalars. Otherwise, np.broadcast(mu, kappa).size samples are drawn.

### **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized von Mises distribution.

#### **See also:**

#### **scipy.stats.vonmises**

probability density function, distribution, or cumulative density function, etc.

#### **Notes**

The probability density for the von Mises distribution is

$$p(x)={\frac{e^{\kappa c o s(x-\mu)}}{2\pi I_{0}(\kappa)}},$$

where *µ* is the mode and *κ* the concentration, and *I*0(*κ*) is the modified Bessel function of order 0.

The von Mises is named for Richard Edler von Mises, who was born in Austria-Hungary, in what is now the Ukraine. He fled to the United States in 1939 and became a professor at Harvard. He worked in probability theory, aerodynamics, fluid mechanics, and philosophy of science.

### **References**

[1], [2]

#### **Examples**

Draw samples from the distribution:

```
>>> mu, kappa = 0.0, 4.0 # mean and concentration
>>> rng = np.random.default_rng()
>>> s = rng.vonmises(mu, kappa, 1000)
```
Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> from scipy.special import i0
>>> plt.hist(s, 50, density=True)
>>> x = np.linspace(-np.pi, np.pi, num=51)
>>> y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))
>>> plt.plot(x, y, linewidth=2, color='r')
>>> plt.show()
```
### method

random.Generator.**wald**(*mean*, *scale*, *size=None*)

Draw samples from a Wald, or inverse Gaussian, distribution.

As the scale approaches infinity, the distribution becomes more like a Gaussian. Some references claim that the Wald is an inverse Gaussian with mean equal to 1, but this is by no means universal.

The inverse Gaussian distribution was first studied in relationship to Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian because there is an inverse relationship between the time to cover a unit distance and distance covered in unit time.

#### **Parameters**

![](_page_186_Figure_1.jpeg)

### **mean**

[float or array_like of floats] Distribution mean, must be > 0.

#### **scale**

[float or array_like of floats] Scale parameter, must be > 0.

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if mean and scale are both scalars. Otherwise, np.broadcast(mean, scale).size samples are drawn.

### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized Wald distribution.

#### **Notes**

The probability density function for the Wald distribution is

$$P(x;mean,scale)={\sqrt{\frac{s c a l e}{2\pi x^{3}}}}e^{\frac{-s c a l e(x-mean)^{2}}{2\cdot m e a n^{2}x}}$$

As noted above the inverse Gaussian distribution first arise from attempts to model Brownian motion. It is also a competitor to the Weibull for use in reliability modeling and modeling stock returns and interest rate processes.

### **References**

[1], [2], [3]

### **Examples**

Draw values from the distribution and plot the histogram:

```
>>> import matplotlib.pyplot as plt
>>> rng = np.random.default_rng()
>>> h = plt.hist(rng.wald(3, 2, 100000), bins=200, density=True)
>>> plt.show()
```
![](_page_187_Figure_6.jpeg)

method

random.Generator.**weibull**(*a*, *size=None*)

Draw samples from a Weibull distribution.

Draw samples from a 1-parameter Weibull distribution with the given shape parameter *a*.

$$X=(-l n(U))^{1/a}$$

Here, U is drawn from the uniform distribution over (0,1].

The more common 2-parameter Weibull, including a scale parameter *λ* is just *X* = *λ*(*−ln*(*U*))1/*a* .

#### **Parameters**

**a**

[float or array_like of floats] Shape parameter of the distribution. Must be nonnegative.

**size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if a is a scalar. Otherwise, np.array(a).size samples are drawn.

#### **Returns**

**out**

[ndarray or scalar] Drawn samples from the parameterized Weibull distribution.

**See also:**

```
scipy.stats.weibull_max
scipy.stats.weibull_min
scipy.stats.genextreme
gumbel
```
#### **Notes**

The Weibull (or Type III asymptotic extreme value distribution for smallest values, SEV Type III, or Rosin-Rammler distribution) is one of a class of Generalized Extreme Value (GEV) distributions used in modeling extreme value problems. This class includes the Gumbel and Frechet distributions.

The probability density for the Weibull distribution is

$$p(x)=\frac{a}{\lambda}(\frac{x}{\lambda})^{a-1}e^{-(x/\lambda)^{a}},$$

where *a* is the shape and *λ* the scale.

The function has its peak (the mode) at *λ*( *a−*1 *a* ) 1/*a* .

When a = 1, the Weibull distribution reduces to the exponential distribution.

#### **References**

[1], [2], [3]

#### **Examples**

Draw samples from the distribution:

**>>>** rng = np.random.default_rng() **>>>** a = 5. *# shape* **>>>** s = rng.weibull(a, 1000)

Display the histogram of the samples, along with the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> def weibull(x, n, a):
... return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
>>> count, bins, _ = plt.hist(rng.weibull(5., 1000))
>>> x = np.linspace(0, 2, 1000)
>>> bin_spacing = np.mean(np.diff(bins))
>>> plt.plot(x, weibull(x, 1., 5.) * bin_spacing * s.size, label='Weibull PDF')
>>> plt.legend()
>>> plt.show()
```
method

![](_page_189_Figure_1.jpeg)

random.Generator.**zipf**(*a*, *size=None*)

Draw samples from a Zipf distribution.

Samples are drawn from a Zipf distribution with specified parameter *a* > 1.

The Zipf distribution (also known as the zeta distribution) is a discrete probability distribution that satisfies Zipf's law: the frequency of an item is inversely proportional to its rank in a frequency table.

### **Parameters**

**a**

[float or array_like of floats] Distribution parameter. Must be greater than 1.

### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if a is a scalar. Otherwise, np.array(a).size samples are drawn.

### **Returns**

### **out**

[ndarray or scalar] Drawn samples from the parameterized Zipf distribution.

### **See also:**

### **scipy.stats.zipf**

probability density function, distribution, or cumulative density function, etc.

### **Notes**

The probability mass function (PMF) for the Zipf distribution is

$$p(k)={\frac{k^{-a}}{\zeta(a)}},$$

for integers *k ≥* 1, where *ζ* is the Riemann Zeta function.

It is named for the American linguist George Kingsley Zipf, who noted that the frequency of any word in a sample of a language is inversely proportional to its rank in the frequency table.

#### **References**

[1]

#### **Examples**

Draw samples from the distribution:

**>>>** a = 4.0 **>>>** n = 20000 **>>>** rng = np.random.default_rng() **>>>** s = rng.zipf(a, size=n)

Display the histogram of the samples, along with the expected histogram based on the probability density function:

```
>>> import matplotlib.pyplot as plt
>>> from scipy.special import zeta
```
*bincount* provides a fast histogram for small integers.

```
>>> count = np.bincount(s)
>>> k = np.arange(1, s.max() + 1)
```

```
>>> plt.bar(k, count[1:], alpha=0.5, label='sample count')
>>> plt.plot(k, n*(k**-a)/zeta(a), 'k.-', alpha=0.5,
... label='expected count')
>>> plt.semilogy()
>>> plt.grid(alpha=0.4)
>>> plt.legend()
>>> plt.title(f'Zipf sample, a={a}, size={n}')
>>> plt.show()
```
#### **Legacy random generation**

The *RandomState* provides access to legacy generators. This generator is considered frozen and will have no further improvements. It is guaranteed to produce the same values as the final point release of NumPy v1.16. These all depend on Box-Muller normals or inverse CDF exponentials or gammas. This class should only be used if it is essential to have randoms that are identical to what would have been produced by previous versions of NumPy.

*RandomState* adds additional information to the state which is required when using Box-Muller normals since these are produced in pairs. It is important to use *RandomState.get_state*, and not the underlying bit generators *state*, when accessing the state so that these extra values are saved.

![](_page_191_Figure_1.jpeg)

Although we provide the *MT19937* BitGenerator for use independent of *RandomState*, note that its default seeding uses *SeedSequence* rather than the legacy seeding algorithm. *RandomState* will use the legacy seeding algorithm. The methods to use the legacy seeding algorithm are currently private as the main reason to use them is just to implement *RandomState*. However, one can reset the state of *MT19937* using the state of the *RandomState*:

```
from numpy.random import MT19937
from numpy.random import RandomState
rs = RandomState(12345)
mt19937 = MT19937()
mt19937.state = rs.get_state()
rs2 = RandomState(mt19937)
# Same output
rs.standard_normal()
rs2.standard_normal()
rs.random()
rs2.random()
rs.standard_exponential()
rs2.standard_exponential()
```
### **class** numpy.random.**RandomState**(*seed=None*)

Container for the slow Mersenne Twister pseudo-random number generator. Consider using a different BitGenerator with the Generator container instead.

*RandomState* and *Generator* expose a number of methods for generating random numbers drawn from a variety of probability distributions. In addition to the distribution-specific arguments, each method takes a keyword argument *size* that defaults to None. If *size* is None, then a single value is generated and returned. If *size* is an integer, then a 1-D array filled with generated values is returned. If *size* is a tuple, then an array with that shape is filled and returned.

### **Compatibility Guarantee**

A fixed bit generator using a fixed seed and a fixed series of calls to 'RandomState' methods using the same parameters will always produce the same results up to roundoff error except when the values were incorrect. *Random-* *State* is effectively frozen and will only receive updates that are required by changes in the internals of Numpy. More substantial changes, including algorithmic improvements, are reserved for *Generator*.

#### **Parameters**

#### **seed**

[{None, int, array_like, BitGenerator}, optional] Random seed used to initialize the pseudorandom number generator or an instantized BitGenerator. If an integer or array, used as a seed for the MT19937 BitGenerator. Values can be any integer between 0 and 2**32 - 1 inclusive, an array (or other sequence) of such integers, or None (the default). If *seed* is None, then the *MT19937* BitGenerator is initialized by reading data from /dev/urandom (or the Windows analogue) if available or seed from the clock otherwise.

### **See also:**

*Generator MT19937 numpy.random.BitGenerator*

#### **Notes**

The Python stdlib module "random" also contains a Mersenne Twister pseudo-random number generator with a number of methods that are similar to the ones available in *RandomState*. *RandomState*, besides being NumPy-aware, has the advantage that it provides a much larger number of probability distributions to choose from.

#### **Seeding and state**

| get_state([legacy]) | Return a tuple representing the internal state of the gen |
| --- | --- |
|  | erator. |
| set_state(state) | Set the internal state of the generator from a tuple. |
| seed([seed]) | Reseed a legacy MT19937 BitGenerator |

#### method

random.RandomState.**get_state**(*legacy=True*)

Return a tuple representing the internal state of the generator.

For more details, see *set_state*.

#### **Parameters**

#### **legacy**

[bool, optional] Flag indicating to return a legacy tuple state when the BitGenerator is MT19937, instead of a dict. Raises ValueError if the underlying bit generator is not an instance of MT19937.

#### **Returns**

#### **out**

[{tuple(str, ndarray of 624 uints, int, int, float), dict}] If legacy is True, the returned tuple has the following items:

- 1. the string 'MT19937'.
- 2. a 1-D array of 624 unsigned integer keys.
- 3. an integer pos.
- 4. an integer has_gauss.
- 5. a float cached_gaussian.

If *legacy* is False, or the BitGenerator is not MT19937, then state is returned as a dictionary.

#### **See also:**

### *set_state*

### **Notes**

*set_state* and *get_state* are not needed to work with any of the random distributions in NumPy. If the internal state is manually altered, the user should know exactly what he/she is doing.

### method

```
random.RandomState.set_state(state)
```
Set the internal state of the generator from a tuple.

For use if one has reason to manually (re-)set the internal state of the bit generator used by the RandomState instance. By default, RandomState uses the "Mersenne Twister"[1] pseudo-random number generating algorithm.

### **Parameters**

### **state**

[{tuple(str, ndarray of 624 uints, int, int, float), dict}] The *state* tuple has the following items:

- 1. the string 'MT19937', specifying the Mersenne Twister algorithm.
- 2. a 1-D array of 624 unsigned integers keys.
- 3. an integer pos.
- 4. an integer has_gauss.
- 5. a float cached_gaussian.

If state is a dictionary, it is directly set using the BitGenerators *state* property.

### **Returns**

### **out**

[None] Returns 'None' on success.

#### **See also:**

#### *get_state*

#### **Notes**

*set_state* and *get_state* are not needed to work with any of the random distributions in NumPy. If the internal state is manually altered, the user should know exactly what he/she is doing.

For backwards compatibility, the form (str, array of 624 uints, int) is also accepted although it is missing some information about the cached Gaussian value: state = ('MT19937', keys, pos).

### **References**

[1]

method

```
random.RandomState.seed(seed=None)
```
Reseed a legacy MT19937 BitGenerator

### **Notes**

This is a convenience, legacy function.

The best practice is to **not** reseed a BitGenerator, rather to recreate a new one. This method is here for legacy reasons. This example demonstrates best practice.

```
>>> from numpy.random import MT19937
>>> from numpy.random import RandomState, SeedSequence
>>> rs = RandomState(MT19937(SeedSequence(123456789)))
# Later, you want to restart the stream
>>> rs = RandomState(MT19937(SeedSequence(987654321)))
```
### **Simple random data**

| rand(d0, d1, ..., dn) | Random values in a given shape. |
| --- | --- |
| randn(d0, d1, ..., dn) | Return a sample (or samples) from the "standard normal" |
|  | distribution. |
| randint(low[, high, size, dtype]) | Return random integers from low (inclusive) to high (ex |
|  | clusive). |
| random_integers(low[, high, size]) | Random integers of type numpy.int_ between low and |
|  | high, inclusive. |
| random_sample([size]) | Return random floats in the half-open interval [0.0, 1.0). |
| choice(a[, size, replace, p]) | Generates a random sample from a given 1-D array |
| bytes(length) | Return random bytes. |

#### method

random.RandomState.**rand**(*d0*, *d1*, *...*, *dn*)

Random values in a given shape.

**Note:** This is a convenience function for users porting code from Matlab, and wraps *random_sample*. That function takes a tuple to specify the size of the output, which is consistent with other NumPy functions like *numpy. zeros* and *numpy.ones*.

Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).

#### **Parameters**

#### **d0, d1, …, dn**

[int, optional] The dimensions of the returned array, must be non-negative. If no argument is given a single Python float is returned.

#### **Returns**

**out**

[ndarray, shape (d0, d1, ..., dn)] Random values.

**See also:**

*random*

### **Examples**

```
>>> np.random.rand(3,2)
array([[ 0.14022471, 0.96360618], #random
      [ 0.37601032, 0.25528411], #random
      [ 0.49313049, 0.94909878]]) #random
```
method

random.RandomState.**randn**(*d0*, *d1*, *...*, *dn*)

Return a sample (or samples) from the "standard normal" distribution.

**Note:** This is a convenience function for users porting code from Matlab, and wraps *standard_normal*. That function takes a tuple to specify the size of the output, which is consistent with other NumPy functions like *numpy.zeros* and *numpy.ones*.

**Note:** New code should use the *standard_normal* method of a *Generator* instance instead; please see the *Quick start*.

If positive int_like arguments are provided, *randn* generates an array of shape (d0, d1, ..., dn), filled with random floats sampled from a univariate "normal" (Gaussian) distribution of mean 0 and variance 1. A single float randomly sampled from the distribution is returned if no argument is provided.

### **Parameters**

```
d0, d1, …, dn
```
[int, optional] The dimensions of the returned array, must be non-negative. If no argument is given a single Python float is returned.

#### **Returns**

#### **Z**

[ndarray or float] A (d0, d1, ..., dn)-shaped array of floating-point samples from the standard normal distribution, or a single such float if no parameters were supplied.

#### **See also:**

#### *standard_normal*

Similar, but takes a tuple as its argument.

### *normal*

Also accepts mu and sigma arguments.

### *random.Generator.standard_normal*

which should be used for new code.

#### **Notes**

For random samples from the normal distribution with mean mu and standard deviation sigma, use:

```
sigma * np.random.randn(...) + mu
```
### **Examples**

```
>>> np.random.randn()
2.1923875335537315 # random
```
Two-by-four array of samples from the normal distribution with mean 3 and standard deviation 2.5:

```
>>> 3 + 2.5 * np.random.randn(2, 4)
array([[-4.49401501, 4.00950034, -1.81814867, 7.29718677], # random
      [ 0.39924804, 4.68456316, 4.99394529, 4.84057254]]) # random
```
method

random.RandomState.**randint**(*low*, *high=None*, *size=None*, *dtype=int*)

Return random integers from *low* (inclusive) to *high* (exclusive).

Return random integers from the "discrete uniform" distribution of the specified dtype in the "half-open" interval [*low*, *high*). If *high* is None (the default), then results are from [0, *low*).

**Note:** New code should use the *integers* method of a *Generator* instance instead; please see the *Quick start*.

#### **Parameters**

#### **low**

[int or array-like of ints] Lowest (signed) integers to be drawn from the distribution (unless high=None, in which case this parameter is one above the *highest* such integer).

#### **high**

[int or array-like of ints, optional] If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior if high=None). If array-like, must contain integer values

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

#### **dtype**

[dtype, optional] Desired dtype of the result. Byteorder must be native. The default value is long.

**Warning:** This function defaults to the C-long dtype, which is 32bit on windows and otherwise 64bit on 64bit platforms (and 32bit on 32bit ones). Since NumPy 2.0, NumPy's default integer is 32bit on 32bit platforms and 64bit on 64bit platforms. Which corresponds to *np.intp*. (*dtype=int* is not the same as in most NumPy functions.)

#### **Returns**

#### **out**

[int or ndarray of ints] *size*-shaped array of random integers from the appropriate distribution, or a single such random int if *size* not provided.

### **See also:**

```
random_integers
```
similar to *randint*, only for the closed interval [*low*, *high*], and 1 is the lowest value if *high* is omitted.

### *random.Generator.integers*

which should be used for new code.

#### **Examples**

```
>>> np.random.randint(2, size=10)
array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
>>> np.random.randint(1, size=10)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```
Generate a 2 x 4 array of ints between 0 and 4, inclusive:

```
>>> np.random.randint(5, size=(2, 4))
array([[4, 0, 2, 1], # random
       [3, 2, 2, 0]])
```
Generate a 1 x 3 array with 3 different upper bounds

**>>>** np.random.randint(1, [3, 5, 10]) array([2, 2, 9]) # random

Generate a 1 by 3 array with 3 different lower bounds

```
>>> np.random.randint([1, 5, 7], 10)
array([9, 8, 7]) # random
```
Generate a 2 by 4 array using broadcasting with dtype of uint8

```
>>> np.random.randint([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
array([[ 8, 6, 9, 7], # random
      [ 1, 16, 9, 12]], dtype=uint8)
```
#### method

random.RandomState.**random_integers**(*low*, *high=None*, *size=None*)

Random integers of type *numpy.int_* between *low* and *high*, inclusive.

Return random integers of type *numpy.int_* from the "discrete uniform" distribution in the closed interval [*low*, *high*]. If *high* is None (the default), then results are from [1, *low*]. The *numpy.int_* type translates to the C long integer type and its precision is platform dependent.

This function has been deprecated. Use randint instead.

Deprecated since version 1.11.0.

#### **Parameters**

#### **low**

[int] Lowest (signed) integer to be drawn from the distribution (unless high=None, in which case this parameter is the *highest* such integer).

#### **high**

[int, optional] If provided, the largest (signed) integer to be drawn from the distribution (see above for behavior if high=None).

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

### **Returns**

### **out**

[int or ndarray of ints] *size*-shaped array of random integers from the appropriate distribution, or a single such random int if *size* not provided.

### **See also:**

#### *randint*

Similar to *random_integers*, only for the half-open interval [*low*, *high*), and 0 is the lowest value if *high* is omitted.

#### **Notes**

To sample from N evenly spaced floating-point numbers between a and b, use:

```
a + (b - a) * (np.random.random_integers(N) - 1) / (N - 1.)
```
#### **Examples**

```
>>> np.random.random_integers(5)
4 # random
>>> type(np.random.random_integers(5))
<class 'numpy.int64'>
>>> np.random.random_integers(5, size=(3,2))
array([[5, 4], # random
       [3, 3],
       [4, 5]])
```
Choose five random numbers from the set of five evenly-spaced numbers between 0 and 2.5, inclusive (*i.e.*, from the set 0*,* 5/8*,* 10/8*,* 15/8*,* 20/8):

```
>>> 2.5 * (np.random.random_integers(5, size=(5,)) - 1) / 4.
array([ 0.625, 1.25 , 0.625, 0.625, 2.5 ]) # random
```
Roll two six sided dice 1000 times and sum the results:

```
>>> d1 = np.random.random_integers(1, 6, 1000)
>>> d2 = np.random.random_integers(1, 6, 1000)
>>> dsums = d1 + d2
```
Display results as a histogram:

![](_page_199_Figure_1.jpeg)

![](_page_199_Figure_2.jpeg)

#### method

```
random.RandomState.random_sample(size=None)
```
Return random floats in the half-open interval [0.0, 1.0).

Results are from the "continuous uniform" distribution over the stated interval. To sample *Unif*[*a, b*)*, b > a* multiply the output of *random_sample* by *(b-a)* and add *a*:

(b - a) * random_sample() + a

```
Note: New code should use the random method of a Generator instance instead; please see the Quick start.
```
#### **Parameters**

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

### **Returns**

### **out**

[float or ndarray of floats] Array of random floats of shape *size* (unless size=None, in which case a single float is returned).

#### **See also:**

#### *random.Generator.random*

which should be used for new code.

### **Examples**

```
>>> np.random.random_sample()
0.47108547995356098 # random
>>> type(np.random.random_sample())
<class 'float'>
>>> np.random.random_sample((5,))
array([ 0.30220482, 0.86820401, 0.1654503 , 0.11659149, 0.54323428]) # random
```
Three-by-two array of random numbers from [-5, 0):

```
>>> 5 * np.random.random_sample((3, 2)) - 5
array([[-3.99149989, -0.52338984], # random
       [-2.99091858, -0.79479508],
       [-1.23204345, -1.75224494]])
```
method

random.RandomState.**choice**(*a*, *size=None*, *replace=True*, *p=None*)

Generates a random sample from a given 1-D array

**Note:** New code should use the *choice* method of a *Generator* instance instead; please see the *Quick start*.

**Warning:** This function uses the C-long dtype, which is 32bit on windows and otherwise 64bit on 64bit platforms (and 32bit on 32bit ones). Since NumPy 2.0, NumPy's default integer is 32bit on 32bit platforms and 64bit on 64bit platforms.

#### **Parameters**

**a**

[1-D array-like or int] If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if it were np.arange(a)

#### **size**

[int or tuple of ints, optional] Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

#### **replace**

[boolean, optional] Whether the sample is with or without replacement. Default is True, meaning that a value of a can be selected multiple times.

**p**

[1-D array-like, optional] The probabilities associated with each entry in a. If not given, the sample assumes a uniform distribution over all entries in a.

### **Returns**

### **samples**

[single item or ndarray] The generated random samples

#### **Raises**

### **ValueError**

If a is an int and less than zero, if a or p are not 1-dimensional, if a is an array-like of size 0, if p is not a vector of probabilities, if a and p have different lengths, or if replace=False and the sample size is greater than the population size

