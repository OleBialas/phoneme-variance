root = fileparts(pwd)
% Extract envelope using Gammachirp filterbank `
%% Set up parameters
wav_fs=44100; % Sampling frequency of source audio files
out_fs=128; % Downsample spectrogram to this frequqency
down_factor=3; % envelope downsampling factor

% Design lowpass filter
Fpass = 9e3;
Fstop = 11e3;
Fs=wav_fs;
Apass = 1;
Astop = 60;
d = designfilt('lowpassiir', 'PassbandFrequency', Fpass, 'StopbandFrequency', Fstop,...
    'StopbandAttenuation', 65, 'SampleRate', wav_fs, 'DesignMethod', 'cheby2');

% GammacHirp filterbank
GCparam.fs = wav_fs/down_factor;
GCparam.FRange = [80,8e3];
GCparam.OutMidCrct = 'ELC';
GCparam.NumCh = 8
Fs = GCparam.fs;

%% Envelope extraction 
folders = char('multi_speakers',' single_speaker');
for f = 1:2
    wavs = dir(strcat('../raw/stimuli/', folders(f, :), '/*.wav'));
    nwavs = length(wavs);
    envelopes={};
    for i = 1:nwavs
        f_name = fullfile(wavs(i).folder, wavs(i).name)
        [y,Fs] = audioread(f_name);
        assert(Fs==wav_fs);
        y=(y(:,1)); % only keep 1st channel
        y = filtfilt(d,y); % Filter below Nyquist frequency
        y= nt_dsample(y,Fs/GCparam.fs);
        Fs = GCparam.fs;
        spectrogram = GCFBv210(y',GCparam); % Bandpass filter
        spectrogram = spectrogram';
        spectrogram = resample(spectrogram, out_fs, Fs);
        Fs=out_fs;

        for chn=1:size(spectrogram,1) % envelope of each spectrogram band
            spectrogram(chn,:)=abs(hilbert(spectrogram(chn,:)));
        end

        save(fullfile('../results/spectrograms/',folders(f,:), strcat(wavs(i).name(1:end-4),'_spg.mat')), 'spectrogram', 'Fs');
    end
end

