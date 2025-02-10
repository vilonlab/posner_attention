# posner_attention
collaboration with Dr Andrew Lynn, using psychopy &amp; quest+

## Setting up your environment:

(assuming you're using anaconda envs and have python installed)
1. Create envionment:
```python
conda create -n psychopy-env python=3.10 
```

> Psychopy requires 3.10, it'll break if you use a newer version of python :)

2. Pip install the necessary packages:
```python
pip install psychopy
pip install numpy
pip install questplus # this might come with psychopy automatically, but I didn't test it.
```

The main file for running the experiment is `Experiment/ViLoN_Posner_Cueing.py`, detailed description of its current state is below.

3. To run `ViLoN_Posner_Cueing.py`, you can open that file in your IDE and select your `psychopy-env` as the interpreter. If you're in VSCode, then you can hit the play/"run" button in the top right corner of the file. 

> There are absolutely other ways to run the file but I think this is the easiest to write down!

4. When you run the experiment, a directory, `data` will be created in the `Experiment` directory of the project. The file name should include a number, the name of the experiment, and then the date time. There are three file outputs: `.csv` (this is the data we'll analyze), `.log` (this is the log of the experiment), and `.psydat` (this is the psychopy data file).

## `Experiment/ViLoN_Posner_Cueing.py` Breakdown:

### Significant sections, Line-by-Line

> 2025-02-10

QuestPlus is initialized for two conditions; lines 48-62

questplus alg is fed into the trial routine in lines 500-515

line 453-457: TrialList is created, basically our conditions architecture

