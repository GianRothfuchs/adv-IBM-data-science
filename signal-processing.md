# Signal Processing
## Discrete Fourier Transformation

### Application
The Fourier transformation decomposes a signal into frequencies that make it up. For example a music chord, can be decomposed in the tree constituting frequencies (i.e. the tones). Each being represented as a sinus wave of the form y(t) = A*sin(2*pi*f*t + p). Where A is the amplitude, f the frequency, and p the phase shift. The follwoing code generates sample waves:

´´´python

	def gen_wave(frequency, amplitude, Time, phaseshift, samplingrate):
		
		time = np.arange(0,T,T/samplingrate)
		X = amplitude * np.sin(2 * np.pi * frequency * Time + phaseshift)
		
		return time,X
		
		
	time, amplitude1 = gen_wave(1, 1, 10, 1, 1000)
	time, amplitude2 = gen_wave(2, 2, 10, 1, 1000)
	time, amplitude3 = gen_wave(3, 4, 10, 0, 1000)

	amplitude = amplitude1 + amplitude2 + amplitude3

´´´

The resulting complex signal is shown below:

![Complex Signal](/screenshots/complex_signal.png)

the complex signal can the be decompsoed again intro its frequencies by aplying Fourier transformation.

´´´python

sr=50 #in Hz

x,y   = gen_wave(1, 1, 10, 1, 50)
x,y2   = gen_wave(2, 2, 10, 1, 50)

y = y + y2

n = len(y) 
p = fft(y) # take the fourier transform 

mag = np.sqrt(p.real**2 + p.imag**2)

mag = mag * 2 / n

mag = mag[0:math.ceil((n)/2.0)]

x = np.arange(0, len(mag), 1.0) * (sr / n)

´´´
Resultign the correct decomposition:

![Complex Signal](/screenshots/complex_signal_decomp.png)


### The Math

Any sampled signal of length N can be represented uniquely and unambiguously by a finite series of sinusoids. This means the generating porcess is not a continuous function but rather a finite dataset x: 
{x} = x_0,x_1,...x_{N-1}, |x| = N

The Fourier transformation turns this into a set X where each element is a tuple (A_k,B_k):
{X} = X_0,X_1,...X_{N-1}
 se the number of sample paoints is driven by the sampling rate as well as the maximum frequency I can observe. Intuitively it is not possible to detect frequencies that are higher as the sampling rate (see Nyquist Point).
 
 the transformed series is obtained by applying the following formula:
 
  X_k =\sum_{n=0}^{N-1} x_n*cos(2*pi*k*n/N) - i*\sum_{n=0}^{N-1} x_n*sin(2*pi*k*n/N)
  
Representing: 
  
  X_k = A_k - i * B_k
  
Note that A_k and B_k are non-complex numbers and therefore representable in ML lib. Unfortuantely there is no ML lib implementation of fourier Transformation, therefore a SystemML implementation has to be created from scratch.

Having the A_k, B_k coefficients the magnitude an phase shift can be calculated in the following way:
Magnitude: |X_k| = sqrt(A_k**2 + B_k**2)
Phase shift: PHI = tan^{-1}(B_k/A_k)

[SystemML](/notebooks/FT.dml) implementation (inefficient,though) fo Fourier Transformation 



