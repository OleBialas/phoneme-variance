rootDir = fileparts(pwd)
% Extract envelope using Gammachirp filterbank `
%% Set up parameters
out_fs=128; % Downsample spectrogram to this frequqency
down_factor=3; % envelope downsampling factor

% Design lowpass filter
Fpass = 9e3;
Fstop = 11e3;
Apass = 1;
Astop = 60;

% GammacHirp filterbank
GCparam.FRange = [80,8e3];
GCparam.OutMidCrct = 'ELC';
GCparam.NumCh = 16;

%% Envelope extraction 
folders = char('multi_speakers','single_speaker');
for f = 1:2
    wavs = dir(fullfile(rootDir, strcat('/raw/stimuli/', folders(f, :), '/*.wav')));
    nwavs = length(wavs);
    envelopes={};
    for i = 1:nwavs
        f_name = fullfile(wavs(i).folder, wavs(i).name)
        [y,Fs] = audioread(f_name);
        
        if f==1 & i==1
            d = designfilt('lowpassiir', 'PassbandFrequency', Fpass,...
                'StopbandFrequency', Fstop, 'StopbandAttenuation', 65, 'SampleRate',...
                Fs, 'DesignMethod', 'cheby2');
            GCparam.fs = Fs/down_factor;
        end

        y=(y(:,1)); % only keep 1st channel
        y = filtfilt(d,y); % Filter below Nyquist frequency
        y= nt_dsample(y,Fs/GCparam.fs);
        Fs = GCparam.fs;
        spectrogram = GCFBv210(y',GCparam); % Bandpass filter
        spectrogram = spectrogram';

        for chn=1:size(spectrogram,1) % envelope of each spectrogram band
            spectrogram(chn,:)=abs(hilbert(spectrogram(chn,:)));
        end

        spectrogram = resample(spectrogram, out_fs, Fs);
        Fs=out_fs;

        save(fullfile(rootDir,'/results/spectrograms/',folders(f,:), strcat(wavs(i).name(1:end-4),'_spg.mat')), 'spectrogram', 'Fs');
    end
end

