sample_rate = 32000
window_size = 1024
hop_size = 500      # So that there are 64 frames per second
mel_bins = 64
fmin = 50       # Hz
fmax = 14000    # Hz

frames_per_second = sample_rate // hop_size
audio_duration = 10     # Audio recordings in DCASE2019 Task5 are all 10 seconds
frames_num = frames_per_second * audio_duration
total_samples = sample_rate * audio_duration

# Fine labels
fine_labels = ['1-1_small-sounding-engine', '1-2_medium-sounding-engine', 
    '1-3_large-sounding-engine', '1-X_engine-of-uncertain-size', 
    '2-1_rock-drill', '2-2_jackhammer', '2-3_hoe-ram', '2-4_pile-driver', 
    '2-X_other-unknown-impact-machinery', '3-1_non-machinery-impact', 
    '4-1_chainsaw', '4-2_small-medium-rotating-saw', '4-3_large-rotating-saw', 
    '4-X_other-unknown-powered-saw', '5-1_car-horn', '5-2_car-alarm', 
    '5-3_siren', '5-4_reverse-beeper', '5-X_other-unknown-alert-signal', 
    '6-1_stationary-music', '6-2_mobile-music', '6-3_ice-cream-truck', 
    '6-X_music-from-uncertain-source', '7-1_person-or-small-group-talking', 
    '7-2_person-or-small-group-shouting', '7-3_large-crowd', 
    '7-4_amplified-speech', '7-X_other-unknown-human-voice', 
    '8-1_dog-barking-whining']
    
fine_classes_num = len(fine_labels)
fine_lb_to_idx = {lb: idx for idx, lb in enumerate(fine_labels)}
fine_idx_to_lb = {idx: lb for idx, lb in enumerate(fine_labels)}
    
# Coarse labels
coarse_labels = ['1_engine', '2_machinery-impact', '3_non-machinery-impact', 
    '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']
    
coarse_classes_num = len(coarse_labels)
coarse_lb_to_idx = {lb: idx for idx, lb in enumerate(coarse_labels)}
coarse_idx_to_lb = {idx: lb for idx, lb in enumerate(coarse_labels)}