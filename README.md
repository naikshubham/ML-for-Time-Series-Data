## ML for Time Series

### A Machine Learning Pipeline
- Feature extraction : what kinds of special features leverage a signal that changes over time?
- Model fitting : what kinds of models are suitable for asking questions with timeseries data?
- Prediction and Validation : How can we validate a model that uses timeseries data? What considerations must we make because it changes in time?

### ML and time series data

#### The Heartbeat Acoustic Data
- Audio is a very common kind of timeseries data. Audio tends to have a very high sampling frequency(often above 20,000 samples per second!)
- Dataset is audio data recorded from the hearts of medical patients. A subset of these patients have heart abnormalities. Can we use only this heartbeat data to detect which subjects have abnormalities?

#### Loading Audio data
- Audio data is often stored in wav files. Each of the files in the dataset contains the audio data for one heartbeat session, as well as the sampling rate for that data.
- Librosa to read audio dataset. It has functions for extracting features, vizualizations and analysis for audio data.
- Data is sored in audio and sampling freq is stored in sfreq. If the sampling freq is 2205, it means there are 2205 samples recorded per second.

```python
from glob import glob
files = glob('data/*.wav')

import librosa as lr
audio, sfreq = lr.load('data/file.wav')
```

#### Inferring time from samples
- **Using only the sampling freq, we can infer the time point of each datapoint in the audio file, relative to the start of the file.**

#### Creating array of timestamps
- To do so, we have two options. The first is to generate a range of indices from zero to the number of datapoints in our audio file, divide each index by the sampling freq, and we have a timepoint for each data point. 
- The second option is to calculate the final timepoint of our audio data using a similar method. Then, use the linspace function to generate evenly-spaced numbers between 0 and the final timepoint.
- In either case, we should have an array of numbers of the same length as our audio data.

```python
# option 1
indices = np.arange(0, len(audio))
time = indices / sfreq

# option 2
final_time = (len(audio) - 1) / sfreq
time = np.linspace(0, final_time, sfreq)
```

#### The New York Stock Exchange dataset
- It runs over a much longer timespan than audio data, and has a sampling freq on the order of one sample per day(compared with 2205 samples per second with the audio data)
- Our goal is to predict the stock value of a company using historical data from the market. As we are predicting a continuos output value,this is a regression problem.

#### Converting a column to a time series
- To ensure that a column within a DataFrame is treated as time series, use the `to_datetime()` function.

```python
df['date'] = pd.to_datetime(df['date'])
```

### Classifying a time series

#### Classification and feature engineering
- Always visualize raw data before fitting models. There is lot of complexity in any machine learning step, and visualizing raw data is important to make sure we know where to begin.
- To plot raw audio, we need 2 things: the raw audio waveform, usually in a 1-d or 2d array. We also need a time point of each sample

```python
ixs = np.arange(audio.shape[-1])
time = ixs / sfreq
fig, ax = plt.subplots()
ax.plot(time, audio)
```

#### What features to use
- Using raw data as input to a classifier is usually too noisy to be useful. An easy first step is to calculate summary statistics of our data, which removes the "time" dimension and gives us a more traditional classification dataset.
- For each time series, we calculate several summary statistics. These then can be used as features for a model. This way we can expand a single feature (raw audio amplitude) to several features (here, the min, max, and average of each sample).

#### Calculating multiple features
- Calculate multiple features for several timeseries. By using the **axis=-1** , we collapse across the last dimension, which is time. The result is an array of numbers, one per timeseries.
- We collapsed a 2-d array into a 1-d array for each feature of interest. We can then combine these as inputs to a model. In the case of classification, we also need a label for each timeseries that allows us to build a classifier.

```python
means = np.mean(audio, axis=-1)
maxs = np.max(audio, axis=-1)
stds = np.std(audio, axis=-1)

print(means.shape)
```

#### Preparing features for scikit learn
- For scikit learn ensure that data has correct shape, which is samples by features.Here we can use the column_stack function, which lets us stack 1-d arrays by turning them into the columns of a 2-d array.
- In addition, the labels array is 1-d, so we need to reshape it so that it is 2-d.

```python
from sklearn.svm import LinearSVC

# reshape to 2-d to work with scikit-learn
X = np.column_stack([means, maxs, stds])
y = labels.reshape([-1, 1])
model = LinearSVC()
model.fit(X, y)
```

#### Scoring scikit-learn model

```python
from sklearn.metrics import accuracy_score

# different input data
predictions = model.predict(X_test)

# score our model with % correct manually
percent_score = sum(predictions == labels_test) / len(labels_test)

# using a sklearn scorer
percent_score = accuracy_score(labels_test, predictions)
```

### Improving features for classification
- Some features that are more unique to timeseries data. Calculate the **envelope** of each heartbeat sound.
- The envelope throws away information about the fine-grained changes in the signal, focusing on the general shape of the audio waveform. To do this we'll need to calculate the audio's amplitude, then smooth it over time.

#### Smoothing over time
- We can remove noise in timeseries data by smoothing it with a rolling window. Instead of averaging over all time, we can do a local average.
- This is called smooting timeseries. This means defining a window around each timepoint, calculating the mean of this window, and then repeating this for each timepoint.

```python
# Calculating a rolling window statistic
window_size=50 # larger the window smoother the result

windowed = audio.rolling(window=window_size)
audio_smooth = windowed.mean()
```

#### Calculating the auditory envelope
- First rectify audio, then smooth it. Calculate the "absolute value" of each timepoint. This is also called "rectification", because we ensure that all time points are positive.
- Next, we calculate a rolling mean to smooth the signal.

```python
audio_rectified = audio.apply(np.abs)
audio_envelope = audio_rectified.rolling(50).mean()
```

##### Feature engineering the envelope
- Once we've calculated the acoustic envelope, we can create better features for our classifier. Here we calculate several common statistics of each auditory envelope, and calculate them in a way that scikit-learn can use.
- Even though we're calculating the same statistics(avg, std-dev and max), they are on different features, and so have different information about the stimulus.

```python
# calculate several features of the envelope, one per sound
envelope_mean = np.mean(audio_envelope, axis=0)
envelope_std = np.std(audio_envelope, axis=0)
envelope_max = np.max(audio_envelope, axis=0)

# create training data for a classifier
X = np.column_stack([envelope_mean, envelope_std, envelope_max])
y = labels.reshape([-1, 1])
```

##### Using cross_val_score

```python
from sklearn.model_selection import cross_val_score

model = LinearSVC()
scores = cross_val_score(model, X, y, cv=3)
print(scores)
```

#### Auditory features : The Tempogram
- There are several more advanced features that can be calculated with timeseries data. Each attempts to detect particular patterns over time, and summarize them statistically.
- For example, a tempogram tells us the tempo of the sound at each moment.

#### Computing the tempogram
- It tells us the moment by moment tempo of the sound. We can then use this to calculate features for our classifier.

```python
# calculate the tempo of 1-d sound array
import librosa as lr
audio_tempo = lr.beat.tempo(audio, sr=sfreq, hop_length=2**6, aggregate=None)
```

### The spectrogram - spectral changes to sound over time

#### Fourier transforms
- Spectograms are common in timeseries analysis. Key part of the spectrograms- the fourier transform.
- This approach summarizes a time series as a collection of fast-and-slow moving waves.
- The Fourier Transform or FFT is a way to tell us how these waves can be combined in different amounts to create our time series. It describes for a window of time, the presence of fast-and-slow-oscillations that are present in a timeseries.
- **The slower oscillations are on the left (closer to 0) and the faster oscillations are on the right**. This is a more rich representation of the audio signal.

### Spectrograms : combination of windows Fourier transforms
- A spectrogram is a collection of windowed Fourier transforms over time.We can calculate multiple fourier transforms in a sliding window to see how it changes over time.
- For each timepoint, we take a window of time around it, calculate a fourier transform for the window, then slide to the next window(similar to calculating the rolling mean).

1. Choose a window size and shape
2. At a timepoint, calculate the FFT for that window
3. Slide the window over by one
4. Aggregate the results
5. Its called a Short-Time Fourier Transform(STFT)

- To calculate the spectogram, we squar each value of the STFT.Note how the spectral content of the sound changes over time.
- Because it is speech, we can see interesting patterns that corresponds to spoken words(e.g vowels or consonants)

#### Calculating the STFT
- Calculate the STFT of the audio file, then convert the output to decibels to visualize it more cleanly with specshow (which results in the visualized spectograms). 

```python
from librosa.core import stft, amplitude_to_db
from librosa.display import specshow

# calculate our STFT
HOP_LENGTH = 2**4
SIZE_WINDOW = 2**7
audio_spec = stft(audio, hop_length=HOP_LENGTH, n_fft=SIZE_WINDOW)

# Convert into decibels for visualization
spec_db = amplitude_to_db(audio_spec) # ensures all values are postive real nos

# visualize
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
```

#### Spectral feature engineering
- Each timeseries has a unique spectral pattern to it. This means we can use patterns in the spectrogram to distinguish classes from one another.
- For example, we can calculate the spectral centroid and bandwidth over time. These describe where most of the spectral energy lies over time.

#### Calculate spectral features

```python
# calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]

# display these features on top of the spectrogram
ax = specshow(spec, x_axis='time', y_axis='hz' hop_length=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=0.5)
```

#### Combining spectral and temporal features in a classifier
- As a final step, we can combine each of the features into a single input matrix for our classifier.

```python
centroids_all = []
bandwidths_all = []

for spec in spectrograms:
  bandwidths = lr.feature.spectral_bandwidth(S=lr.db_to_amplitude(spec))
  centroids = lr.feature.spectral_centroids(S=lr.db_to_amplitude(spec))
  # calculate the mean spectral bandwidth
  bandwidths_all.append(np.mean(bandwidths))
  # calculate the mean spectral centroid
  centroids_all.append(np.mean(centroids))
  
# create our X matrix
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths_all, centroids_all])
```

### Predicting data over time
- Now focus on regression. Regression has several features and caveats that are unique to timeseries data.

#### Correlation and Regression
- Both regression and correlation reflect the extent to which the values of two variables have a consistent relationship(either they both go down or up together, or they have an inverse relationship).
- However regression results in a "model" of the data, while correlation is just a single statistics that describes the data. Regression models have more information about the data, while correlation is easier to calculate and interpret.

#### Correlation between variables often changes over time
- When running regression models with timeseries data, it's important to visualize how the data changes over time. We can either do this by plotting the whole timeseries at once,or by directly comparing two segments of time.

#### Visualizing relationships between timeseries

```python
fig, axs = plt.subplots(1, 2)

# Make a line plot for each timeseries
axs[0].plot(x, c='k', lw=3, alpha=.2)
axs[0].plot(y)
axs[0].set(xlabel='time', title='X values = time')

# encode time as color in a scatterplot
axs[1].scatter(x_long, y_long, c=np.arange(len(x_long)), cmap='viridis')
axs[1].set(xlabel='x', ylabel='y', title='Color = time')
```

#### Regression models with scikit-learn

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
model.predict(X)
```

#### Visualize predictions with scikit-learn
- We can visualize the predictions from several different models fit on the same data. We'll use Ridge Regression, which has parameter called "alpha" that causes coefficients to be smoother and smaller, and is useful if we have noisy or correlated variables.
- Loop through a few values of alpha, initializing a model with each one and fitting it on the training data. Plot model predictions on the test data.

```python
alphas=[.1, 1e2, 1e3]
ax.plot(y_test, color='k', alpha=.3, lw=3)
for ii, alpha in enumerate(alphas):
    y_predicted = Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)
    ax.plot(y_predicted, c=cmap(ii / len(alphas))
ax.legend(['True Values', 'Model 1', 'Model 2', 'Model 3'])
ax.set(xlabel="Time")
```

### Advanced time series predictions

#### Cleaning and improving our data
- Real world data is always messy and requires preparing and cleaning the data before fitting models. In timeseries, messsy data often happens due to failing sensors or human error in logging the data.
- **Note that it is not always clear whether patterns in the data are "aberrations" or not. We should always investigate to understand the source of strange patterns in the data.**

#### Interpolation : using time to fill in missing data
- Fill in the missing data using other datapoints we do have. We can use a technique called interpolation, which uses the values on either end of a missing window of time to infer what's in-between.

#### Interpolation in Pandas

```python
# return a boolean that notes where missing values are
missing = prices.isna()

# interpolate linearly within missing windows
prices_interp = prices.interpolate('linear')

# plot the interpolation data in read and the data with missing values in black
ax = prices_interp.plot(c='r')
prices.plot(c='k', ax=ax, lw=2)
```

#### Using a rolling window to transform data
- Another common technique to clean data is transforming it so that it is more well behaved. To do this we use the rolling window technique.
- Using a rolling window, we calculate each timepoint's percent change over the mean of a window of previous timepoints. This standardizes the variance of our data and reduces long-term drift.

#### Transforming to percent change with Pandas

```python
def percent_change(values):
    """Calculate the % change between the last value and the mean of previous values"""
    # separate the last value and all prev values into variables
    previous_values = values[:-1]
    last_value = values[-1]
    
    # calculate the % diff between the last val and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return precent_change
```

- Applying this to the data using the `.aggregate` method, passing our function as an input. Then the data is roughly centered to zero, and periods of high and low changes are easier to spot.

```python
# plot the raw data
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = prices.plot(ax=axs[0])

# calculate % change and plot
ax = prices.rolling(window=20).aggregate(percent_change).plot(ax=axs[1])
ax.legend_.set_visible(False)
```

#### Finding outliers in our data
- We will use this transformation to detect outliers. Outliers are datapoints that are significantly statistically different from the dataset.
- A common definition is any datapoint that is more than 3 standard deviations away from the mean of the dataset.

#### Plotting a threshold on data : Visualize outliers
- Calculate the mean and std of each dataset, then plot outlier "thresholds"(3 * sd from mean) on the raw and transformed data.

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for data, ax in zip([prices, prices_perc_change], axs):
    # calc mean/std for the data
    this_mean = data.mean()
    this_std = data.std()
    
    # Plot the data, with a window that is 3 stds around the mean
    data.plot(ax=ax)
    ax.axhline(this_mean + this_std * 3, ls='--', c='r')
    ax.axhline(this_mean - this_std * 3, ls='--', c='r')
```

<p align="center">
  <img src="data/outliers.JPG" width="350" title="Outliers">
</p>

- Any datapoints outside these bounds could be an outlier. **Note that the datapoints deemed an outlier depend on the transformation of the data. On the right we see a few outlier datapoints that were not outliers in the raw data.**

#### Replacing outliers using the threshold
- Replace outliers with the median of the remaining values. We first center the data by subtracting its mean, and calculate the standard dev. Finally, we calculat e the abs value of each datapoint, and mark any that lie outside of 3 std from the mean.
- We then replace these using the nanmedian function, which calculates the median without being hindered by missing values.

```python
# center the data so the mean is 0
prices_outlier_centered = prices_outlier_perc - prices_outlier_perc.mean()

# calc standard dev
std = prices_outlier_perc.std()

# use the abs val of each data point to make it easier to find outliers
outliers = np.abs(prices_outlier_centered) > (std * 3)

# Repalce outliers with the median value
# We'll use np.nanmean since there may be nans around the outliers
prices_outlier_fixed = prices_outlier_centered.copy()
prices_outlier_fixed[outliers] = np.nanmedian(prices_outlier_fixed)
```

#### Visualize the results

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
prices_outlier_centered.plot(ax=axs[0])
prices_outlier_fixed.plot(ax=axs[1])
```






















