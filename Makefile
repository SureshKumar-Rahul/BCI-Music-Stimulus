.PHONY: play analyze plots help

help:
	@echo "make play      open the music player + EEG recorder GUI"
	@echo "make analyze   run the ANN feature-extraction / training / evaluation script"
	@echo "make plots     batch-plot the recorded EEG signals"

play:
	python run.py play

analyze:
	python run.py analyze

plots:
	python run.py plots
