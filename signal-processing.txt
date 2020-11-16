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

### The Math

Any sampled signal of length N can be represented uniquely and unambiguously by a finite series of sinusoids. This means the generating porcess is not a continuous function but rather a finite dataset x: 
{x} = x_0,x_1,...x_{N-1}, |x| = N

The Fourier transformation turns this into a set X where each element is a tuple (A_k,B_k):
{X} = X_0,X_1,...X_{N-1}
 se the number of sample paoints is driven by the sampling rate as well as the maximum frequency I can observe. Intuitively it is not possible to detect frequencies that are higher as the sampling rate (see Nyquist Point).
 
 the transformed series is obtained by applying the following formula:
 
  X_k =\sum_{n=0}^{N-1} x_n*cos(2*pi*k*n/N) - i*\sum_{n=0}^{N-1} *sin(2*pi*k*n/N)
  
Representing: 
  
  X_k = A_k - i * B_k
  
Note that A_k and B_k are non-complex numbers and therefore representable in ML lib.
