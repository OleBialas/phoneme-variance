% Extract envelope using Gammachirp filterbank 
%% Set up parameters
wav_fs=44100; % Sampling frequency of source audio files
down_factor=10; % envelope downsampling factor

% Parameters for lowpass filter
Fpass = 9e3;
Fstop = 11e3;
Fs=wav_fs;
Apass = 1;
Astop = 60;
h = fdesign.lowpass(Fpass,Fstop,Apass,Astop,Fs);
lpf1 = design(h,'cheby2','MatchExactly','stopband');
clear Fpass Fstop Fs h;

% GammaChirp filterbank
GCparam.fs = wav_fs/down_factor; 
GCparam.NumCh = 8;
GCparam.FRange = [80,8e3];
GCparam.OutMidCrct = 'ELC';
% GCparam.OutMidCrct = 'No';
% GCparam.Ctrl = 'dyn';

% Envelope filter
Fpass = 30;
Fstop = 32;
Fs = GCparam.fs;
h = fdesign.lowpass(Fpass,Fstop,Apass,Astop,Fs);
lpf2 = design(h,'cheby2','MatchExactly','stopband');
clear Fpass Fstop Fs h;

%% Envelope extraction 
folders = char('multi_speakers','single_speaker');
for f = 1:2
    wavs = dir(strcat('../raw/stimuli/', folders(f, :), '/*.wav'));
    nwavs = length(wavs);
    envelopes={};
    for i = 1:nwavs

        % Read in audio
        f_name = fullfile(wavs(i).folder, wavs(i).name)
        [y,Fs] = audioread(f_name);
        assert(Fs==wav_fs);
        y=(y(:,1)); % only keep 1st channel
        y = filtfilthd(lpf1,y); % Filter below Nyquist frequency
        y= nt_dsample(y,Fs/GCparam.fs); % Downsample
        Fs = GCparam.fs
        y=y';
        spectrogram = GCFBv210(y,GCparam); % Bandpass filter

        for chn=1:size(spectrogram,1) % envelope of each spectrogram band
            spectrogram(chn,:)=abs(hilbert(spectrogram(chn,:)));
        end

        save(fullfile('../results/spectrograms/',folders(f,:), strcat(wavs(i).name(1:end-4),'_spg.mat')), 'spectrogram', 'Fs');
    end
end

