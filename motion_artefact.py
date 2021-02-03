import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.optimize import curve_fit 
from tqdm import tqdm 
folder ="C:/motion_artefact"



def cross_corr(y1, y2):
  """Calculates the cross correlation and lags without normalization.

  The definition of the discrete cross-correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html

  Args:
    y1, y2: Should have the same length.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """
  if len(y1) != len(y2):
    raise ValueError('The lengths of the inputs should be the same.')

  y1_auto_corr = np.dot(y1, y1) / len(y1)
  y2_auto_corr = np.dot(y2, y2) / len(y1)
  corr = np.correlate(y1, y2, mode='same')
  # The unbiased sample size is N - lag.
  unbiased_sample_size = np.correlate(
      np.ones(len(y1)), np.ones(len(y1)), mode='same')
  corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
  shift = len(y1) // 2

  max_corr = np.max(corr)
  argmax_corr = np.argmax(corr)
  return max_corr, argmax_corr - shift






def shift(value, array):
    """
    Shift the values of the array to a given lag.
    Args: 
        value: lag 
        array: array to shift, must be 1D numpy array 
        
    Returns : 
        Shifted array with value lag. 
    """
    if value >0: 
        return np.concatenate([array[len(array) - value:], array[:len(array) - value]])
    elif value ==0: 
        return array
    else: 
        return np.concatenate([array[abs(value):] ,array[:abs(value)]]) 
        
    
def find_max_corr(y1,y2): 
    assert len(y1) == len(y2)
    corrs = []
    for i in range(len(y1)): 
        corrs.append(float(np.correlate(y1, shift(i, y2))))
                         
    return corrs, shift(np.argmax(corrs), y2)

def load_data(folder, plot = True) :
    df = pd.read_csv(folder, header = None)
    
    if plot: 
        fig, axes = plt.subplots(2)
        axes[0].plot(np.array((df.apply(np.mean, axis = 1))))
        axes[1].imshow(df.T, cmap = "prism", aspect = "auto")
    return df


def align_traces(data, region, baseline, plot = True):
    df = data.iloc[region[0]: region[1], :] 
    baseline = df.iloc[:, baseline[0]: baseline[1]].apply(np.mean, axis = 1)
    print(baseline.shape)
    print(df.shape)
    df_shifted = pd.DataFrame()
    for i in tqdm(range(df.shape[1])): 
        max_corr, lag = find_max_corr(baseline,df.iloc[:, i])
        lag = np.array(lag)
        df_shifted = df_shifted.append(pd.DataFrame(lag).T)
    
    df_shifted = df_shifted.T
    df_shifted.columns = range(df_shifted.shape[1])
        
    if plot: 
        fig, axes = plt.subplots(2)
        axes[0].imshow(df, cmap = "gist_ncar", aspect = "auto")
        axes[1].imshow(df_shifted, cmap = "gist_ncar", aspect = "auto")
        
    
    return df_shifted





def get_spikes(trace, thr = 3):
    
    mean_trace = trace.apply(np.mean)
    return mean_trace > np.mean(mean_trace) + (thr * np.std(mean_trace))


def extract_spike(trace, bool_mask, sensitivity = 5): 
    
    mean_trace = trace.apply(np.mean)
    for i in range(len(mean_trace)):
        if all(bool_mask[i:i+5]): 
            return np.array(mean_trace[i-50: i+200])

def exp_dec(x, a, b, c): 
    return a * np.exp(-b *x) + c
def exp(x, a, b,c): 
    return a * np.exp(b *x) + c


def fit_exponential(spike): 

    x_data_dec = np.linspace(0, 1, len(spike) - np.argmax(spike))
    y_data_dec = spike[np.argmax(spike):]
    popt, pcov = curve_fit(exp_dec, x_data_dec, y_data_dec)
    
    
    x_data_asc = np.linspace(0,1, np.argmax(spike))
    y_data_asc = spike[:np.argmax(spike)]
    popt_asc, pcov_asc = curve_fit(exp, x_data_asc, y_data_asc)
    
    
    fig, axes = plt.subplots(2)
    axes[0].plot(x_data_asc, y_data_asc)
    axes[0].plot(x_data_asc, exp(x_data_asc, *popt_asc), "r--")
    axes[1].plot(x_data_dec, y_data_dec)
    axes[1].plot(x_data_dec, exp_dec(x_data_dec, *popt), "r--")
    
    
    return np.concatenate([exp(x_data_asc, *popt_asc), exp_dec(x_data_dec, *popt)])





df = load_data(os.path.join(folder, "trace2.csv"))
test = align_traces(df,[0, 100], [30, 150])
bool_mask = get_spikes(test)
spike = extract_spike(test, bool_mask)
plt.plot(spike)
fit = fit_exponential(spike)
plt.plot(spike, "b", alpha = 0.5)
plt.plot(fit, "r--", linewidth=4)
    