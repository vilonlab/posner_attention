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

*Line 38*: Initialize QuestPlus Algorithm

*Line 45* takes establishes the "outcome domain", when I had been previously testing the algorithm/experiment, when I got the answer wrong (incorrectly identified the gabor stripes as vertical or horizontal), the contrast of the gabor would (not as intended) get weaker, thereby making the task harder. This is the opposite of what we want out of questplus - so I tried flipping the response values and I got the intended result! I would like to know more of "why" this was the fix, which we should be able to answer by looking carefully at the `outcome_domain` input argument for the `QuestPlus` class, but this is a good enough to move on building out the experiment.

*Lines 380-384,* parameters for the stimuli, "TDW hardcoded values"
> Note: any "hardcoded" values I implement have all caps variable names.

I wanted to make it so that we could change the size, position, and spatial frequency of the gabor stimuli by just changing these values.

*Lines 386-413* are where these parameters are implemented into the experiment. 

## Next Steps

- Implement different conditions
- Intermix the trials for the conditions
- Test to make sure the algorithm is tracking as we intend; need to talk to Andrew about this. E.g., should the algorithm be tracking each condition independently from one another, or should the accuracy of the participant be tracked across both conditions? 
- Task: Outline the experimental design within this readme.
