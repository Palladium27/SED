# SED
Parameter tuning for wavelet based sound event detection

Wavelet-based audio processing is used for sound event detection. The
low-level audio features (timbral or temporal features) are found to be effective
to differentiate between different sound event and that is why frequency processing
algorithms have become popular in recent times. Wavelet based sound
event detection is found effective to detect sudden onsets in audio signals because
it offers unique advantages compared to traditional frequency-based sound event
detection using CNN’s or RNN’s. In this work, wavelet transform is applied to
the audio to extract audio features which can predict the occurrence of a sound
event using a classical feedforward neural network. Additionally, this work attempts
to identify the optimal wavelet parameters to enhance classification performance.
3 window sizes, 6 wavelet families, 4 wavelet levels, 3 decomposition
levels and 2 classifier models are used for experimental analysis. The UrbanSound8k
data is used and a classification accuracy up to 97% is obtained.
Some major observations with regard to parameter-estimation are as follows:
wavelet level and wavelet decomposition level should be low; it is desirable to
have a large window; however, the window size is limited by the duration of the
sound event. A window size greater than the duration of the sound event will
decrease classification performance. All wavelet family can classify the sound
events; however, using Symlet, Daubechies, Reverse biorthogonal and Biorthogonal
families will save computational resources (lesser epochs) because they
yield better accuracy compared to Fejér-Korovkin and Coiflets. This work conveys
that wavelet-based sound event detection seems promising, and can be expanded
to detect most of the common sounds and sudden events occurring at
various environments.

The work was presented at the conference evoMUSART 2021, and here's the link for it: https://link.springer.com/chapter/10.1007/978-3-030-72914-1_16
