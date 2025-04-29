# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'sounddevice'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

from questplus import QuestPlus

####### BACKEND SETUP ####################################################################################################################################################################################################

exp_name = 'Enhancement_Task'
exp_info = {
    'Participant ID': '',
    'Session': '',
}

dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name)
if dlg.OK == False:
    core.quit()

_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
filename = os.path.join(_thisDir, f"data/{exp_info['Participant ID']}_{exp_name}")

# create an experiment handler to help with data saving
thisExp = data.ExperimentHandler(
    name=exp_name, version='',
    extraInfo=exp_info, runtimeInfo=None,
    originPath='/Users/trentonwirth/GitHub/posner_attention/Experiment/ViLoN_Posner_Cueing.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename, 
    sortColumns='time')

logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag to indicate if the experiment is running
frameTolerance = 0.001  # how close to onset before 'same' frame

# Window setup for EIZO monitor
win = visual.Window(fullscr=True, color=[0,0,0],
            size=[1512, 982], screen=0,
            winType='pyglet', allowStencil=False,
            monitor='Eizo', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False)

# Store frame rate
exp_info['frameRate'] = win.getActualFrameRate()
if exp_info['frameRate'] is not None:
    frameDur = 1.0 / round(exp_info['frameRate'])
else:
    frameDur = 1.0 / 60.0 # couldn't get a reliable measure so guess
    logging.warning('Frame rate is unknown. Using frame duration of 1/60s.')
exp_info['frameDur'] = frameDur    

# Device setup
ioConfig = {
    'Keyboard': dict(use_keymap='psychopy'),
    # 'eyetracker.hw.sr_research.eyelink.EyeTracker': {...}  # ADD EYETRACKER LATER
}

# Launch iohub server to keep track of devices
ioServer = io.launchHubServer(window=win, **ioConfig)
deviceManager = hardware.DeviceManager()
deviceManager.ioServer = ioServer

# Add and get defaultKeyboard
if deviceManager.getDevice('defaultKeyboard') is None:
    deviceManager.addDevice(
        deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
    )
defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    
####### QUESTPLUS INITIALIZATION ####################################################################################################################################################################################################

stim_domain = {'intensity': np.arange(0.01, 1, 0.01)}
param_domain = {
    'threshold': np.arange(0.01, 1, 0.01),
    'slope': np.arange(2, 5, 1),
    'lower_asymptote': 0.5, # Equal to chance
    'lapse_rate': np.arange(0, 0.05, 0.01) # Test 0:0.05 for adults, Consider 0:0.10 for children
}
outcome_domain = {'response': [1,0]}  # I'm going to flip this, to see if it fixes the way I intuitively think the algorithm should work; TDW 2025-01-22

# *THREE* QuestPlus staircases - one for each condition
qp_valid = QuestPlus(
    stim_domain=stim_domain,
    param_domain=param_domain,
    outcome_domain=outcome_domain,
    func='weibull',
    stim_scale='linear'
)

qp_invalid = QuestPlus(
    stim_domain=stim_domain,
    param_domain=param_domain,
    outcome_domain=outcome_domain,
    func='weibull',
    stim_scale='linear'
)

qp_neutral = QuestPlus(
    stim_domain=stim_domain,
    param_domain=param_domain,
    outcome_domain=outcome_domain,
    func='weibull',
    stim_scale='linear'
)

####### EXPERIMENT PARAMETERS ####################################################################################################################################################################################################

# Create a dictionary of 12 unique trial types used to create practice and experiment trial lists
trial_types = data.createFactorialTrialList({
            'orientation': [0, 90], # 0 - vertical; 90 - horizontal
            'gabor_position': [-1, 1], # -1 = Left, 1 = Right
            'cue_condition': ['Neutral', 'Invalid','Valid']  
            }) 

TRIAL_REPETITIONS = 16 # How many times to repeat each of the 12 unique trial types ( Total # trials = TRIAL_REPEITIONS * 12)
PRACTRIALS_REPETITIONS = 1 # Same as above, but for practice trials
MAX_REPEATS = 3 # Maximum number of consecutive trials with the same cue condition
CUE_SIZE = [.5, .5]
TARGET_SIZE = [1.5, 1.5]
FIXATION_SIZE = [.5, .5]
POSITION = np.array([5.0, 0.0]) # 5DVA eccentricity 
SPATIAL_FREQUENCY = 5
PRACT_CONTRASTS = [0.1, 0.5, 1.0] * 4  # Contrast values for practice trials, length equals trial_types length

# Timing (s)
ITI = 1.0 # fixation point between trials
CUE_DURATION = 0.05 
ISI = 0.1 # fixation point between cue and target
TARGET_DURATION = 1.0 
RESPONSE_DURATION = 1.0 # total response window is TARGET_DURATION + RESPONSE_DURATION 
TOTAL_TRIAL_DURATION = ITI + CUE_DURATION + ISI + TARGET_DURATION + RESPONSE_DURATION
BREAK_INTERVAL = 32 # Number of trials before each break
MAX_PRESENTATIONS = 3 # Maximum number of times each trial can be presented 

####### INITIALIZE VISUALS #################################################################################################################################################################################################### 

# --- Initialize components for Instructions ---
Instruct_Text = visual.TextStim(win=win, name='mainInst',
    text='''Welcome to the Line Grate Game!\n\n\n In this game, you will see line grates like these:\n\n\n\n
    Your job is to determine which way the lines are pointing!\n\nPress the left button if the lines are pointing up and down, 
    and the right button if the lines are pointing side to side.''',
    font='Arial',
    units='height', pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
Gabor_Inst1 = visual.GratingStim(
    win=win, name='Gabor',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=90, pos=[5,0], size=[4,4], sf=[SPATIAL_FREQUENCY], phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
Gabor_Inst2 = visual.GratingStim(
    win=win, name='Gabor',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=0.0, pos=[-5,0], size=[4,4], sf=[SPATIAL_FREQUENCY], phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)

# --- Initialize components for Trials ---
Fixation_Point = visual.ShapeStim(
    win=win, name='Fixation_Point', vertices='cross',units='deg', 
    size=(FIXATION_SIZE[0], FIXATION_SIZE[1]),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=0.0, interpolate=True)
Left_Cue = visual.ShapeStim(
    win=win, name='Left_Cue',units='deg', 
    size=[CUE_SIZE[0], CUE_SIZE[1]], vertices='circle',
    ori=0.0, pos=[-POSITION[0], POSITION[1]], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=-1.0, interpolate=True)
Right_Cue = visual.ShapeStim(
    win=win, name='Right_Cue',units='deg', 
    size=[CUE_SIZE[0], CUE_SIZE[1]], vertices='circle',
    ori=0.0, pos=[POSITION[0], POSITION[1]], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=-1.0, interpolate=True)
Gabor = visual.GratingStim(
    win=win, name='Gabor',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=0.0, pos=[POSITION[0],POSITION[1]], size=[TARGET_SIZE[0], TARGET_SIZE[1]], sf=[SPATIAL_FREQUENCY], phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='add',
    texRes=128.0, interpolate=True)
feedback = visual.TextStim(win=win, name='feedback',
    text="",
    font='Arial',
    units='height', pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
feedbackImage = visual.ImageStim(
    win=win, name='feedbackImage',units='deg', 
    image='sin', mask=None,
    ori=0.0, pos=(4, 0), anchor='center',
    size=[0.2,0.2], color=[1,1,1], colorSpace='rgb',
    opacity=1.0, flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
break_text = visual.TextStim(win=win, name='break_text',
    text="Great job!\nLet's take a quick break!", font ='Arial', color= 'black',
    units='height', pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0)
key_resp = keyboard.Keyboard()

# --- Initialize components for the End ---
end_text = visual.TextStim(win=win, name='end_text',
    text="You finished the game!",
    font='Arial',
    pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);


####### INSTRUCTIONS  #################################################################################################################################################################################################### 

# Set clocks and start experiment
globalClock = core.Clock()
routineTimer = core.Clock()
logging.setDefaultClock(globalClock)
win.flip()

exp_info['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
thisExp.status = STARTED

Instruct_Text.draw()
Gabor_Inst1.draw()
Gabor_Inst2.draw()
win.flip()
thisExp.addData('Instructions.started', globalClock.getTime(format='float'))

if defaultKeyboard.getKeys(keyList=["escape"]):
    thisExp.status = FINISHED
    win.close()
    core.quit()

event.waitKeys(keyList=['space']) #Press space to continue to practice trials
thisExp.addData('Instructions.stopped', globalClock.getTime(format='float'))
thisExp.nextEntry()

####### FUNCTIONS #################################################################################################################################################################################################### 

# Determine the opacity (location) of the cue based on the trial's cue condition
def get_cue_opacity(cue_condition, gabor_position):
    if cue_condition == 'Neutral': # show both cues
        return 1.0, 1.0 
    elif cue_condition == 'Valid': # show cue in same position as target gabor
        if gabor_position == -1:
            return 1.0, 0.0 
        else:
            return 0.0, 1.0
    elif cue_condition == 'Invalid': # show cue in opposite position as target gabor
        if gabor_position == 1:
            return 1.0, 0.0
        else:
            return 0.0, 1.0
        
# Check that trial list does not have more than MAX_REPEATS consecutive trials with the same cue condition
def consecutive_check(all_trials, max_repeats=MAX_REPEATS):
    consecutive_count = 1
    if len(all_trials) < 2:
        return False
    for i in range(1, len(all_trials)):
        if all_trials[i]['cue_condition'] == all_trials[i - 1]['cue_condition']:
            consecutive_count += 1
            if consecutive_count > max_repeats:
                return False  # Invalid trial list
        else:
            consecutive_count = 1
    return True  # Valid trial list
        
# Get the full list of trials created by the handler
def get_trials(repetitions):
    trial_list = []
    while consecutive_check(trial_list) == False:
        handler = data.TrialHandler(nReps=repetitions, method='random', 
            extraInfo=exp_info, originPath=-1,
            trialList=trial_types,
            seed=None, name='handler')
        trial_sequence = handler.sequenceIndices 
        trial_indices = trial_sequence.T.flatten().tolist()
        trial_list = [dict(handler.trialList[i]) for i in trial_indices] 
        for i, trial in enumerate(trial_list, start=1):
            trial['Index'] = i
    return trial_list

def draw_comp(comp, t, tThisFlipGlobal, frameN):
    comp.frameNStart = frameN
    comp.tStart = t
    comp.tStartRefresh = tThisFlipGlobal
    win.timeOnFlip(comp, 'tStartRefresh')
    comp.status = STARTED
    if not isinstance(comp, keyboard.Keyboard):  # key_resp is handled differently
        thisExp.timestampOnFlip(win, f'{comp.name}.started')
        comp.setAutoDraw(True)
    else:
        thisExp.timestampOnFlip(win, 'key_resp.started')

def erase_comp(comp, t, tThisFlipGlobal, frameN):
    comp.tStop = t
    comp.tStopRefresh = tThisFlipGlobal
    comp.frameNStop = frameN
    comp.status = FINISHED
    if not isinstance(comp, keyboard.Keyboard):
        thisExp.timestampOnFlip(win, f'{comp.name}.stopped')
        comp.setAutoDraw(False)
    else:
        thisExp.timestampOnFlip(win, 'key_resp.stopped')


def run_trial(trial, practice = False):
    continueRoutine = True
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []

    components = [Fixation_Point, Left_Cue, Right_Cue, Gabor, key_resp]

    # Reset status of components
    for comp in components:
        comp.tStart = None
        comp.tStop = None
        comp.tStartRefresh = None
        comp.tStopRefresh = None
        if hasattr(comp, 'status'):
            comp.status = NOT_STARTED

    # Reset timers
    t = 0
    frameN = -1

    Gabor.pos = np.array([POSITION[0] * trial['gabor_position'], POSITION[1]])
    Gabor.ori = trial['orientation']
    Left_Cue.opacity, Right_Cue.opacity = get_cue_opacity(trial['cue_condition'], trial['gabor_position'])

    if practice: # Set Gabor contrast and orientation for practice trials
        intensity = pract_contrasts.pop(0)
        Gabor.contrast = intensity
    
    if not practice: # Experiment trials, questplus algorithm will determine Gabor contrast
        trial['presented'] += 1
        if trial['cue_condition'] == 'Valid':
            current_qp = qp_valid
        elif trial['cue_condition'] == 'Invalid':
            current_qp = qp_invalid
        elif trial['cue_condition'] == 'Neutral':
            current_qp = qp_neutral  
        
        # Save threshold, slope, and lapse rate from QP to data file
        threshold = current_qp.param_estimate['threshold']
        slope = current_qp.param_estimate['slope']
        lapse_rate = current_qp.param_estimate['lapse_rate']

        # Get next intensity from current staircase
        next_stim = current_qp.next_stim
        intensity = next_stim['intensity']
        print(f"Orientation: {trial['orientation']}, Next Intensity: {intensity}, Cue condition: {trial['cue_condition']}")
        
        # Update Gabor contrast
        Gabor.contrast = intensity

    while continueRoutine: # to update components on each frame
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  
    
        if Fixation_Point.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            Fixation_Point.color = 'white'
            draw_comp(Fixation_Point, t, tThisFlipGlobal, frameN)
        
        if Fixation_Point.status == STARTED:
            if tThisFlip >= 0.9-frameTolerance and tThisFlip < ITI-frameTolerance:
                Fixation_Point.color = 'green'
            if tThisFlipGlobal > Fixation_Point.tStartRefresh + TOTAL_TRIAL_DURATION-frameTolerance:
                erase_comp(Fixation_Point, t, tThisFlipGlobal, frameN)
        
        if Left_Cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
            draw_comp(Left_Cue, t, tThisFlipGlobal, frameN)
            draw_comp(Right_Cue, t, tThisFlipGlobal, frameN)
            Fixation_Point.color = 'white'
        
        if Left_Cue.status == STARTED and tThisFlipGlobal > Left_Cue.tStartRefresh + CUE_DURATION-frameTolerance:
            erase_comp(Left_Cue, t, tThisFlipGlobal, frameN)
            erase_comp(Right_Cue, t, tThisFlipGlobal, frameN)

        if Gabor.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
            draw_comp(Gabor, t, tThisFlipGlobal, frameN)

        if Gabor.status == STARTED and tThisFlipGlobal > Gabor.tStartRefresh + TARGET_DURATION-frameTolerance:
            erase_comp(Gabor, t, tThisFlipGlobal, frameN)
            Fixation_Point.color = 'blue'
        
        waitOnFlip = False  # Wait for key response

        if key_resp.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
            draw_comp(key_resp, t, tThisFlipGlobal, frameN)
            waitOnFlip = True  # Wait for key response
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')
        
        if key_resp.status == STARTED and tThisFlipGlobal > key_resp.tStartRefresh + (TARGET_DURATION+RESPONSE_DURATION)-frameTolerance:
            erase_comp(key_resp, t, tThisFlipGlobal, frameN)

        if key_resp.status == STARTED and not waitOnFlip: 
            theseKeys = key_resp.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                thisExp.timestampOnFlip(win, 'key_resp.stopped') # record time of key press
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                continueRoutine = False  # end the routine on key press
    
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
            win.close()
            core.quit()

        if continueRoutine: # To ensure that trial ends when a key is pressed unless a component is still running
            continueRoutine = False
            for comp in components:
                if hasattr(comp, 'status') and comp.status != FINISHED:
                    continueRoutine = True
                    break
        
        if continueRoutine:
            win.flip()
    
    for comp in components:
        if hasattr(comp, 'setAutoDraw'):
            comp.setAutoDraw(False)
            
    # Add trial data to the data file
    thisExp.addData('Gabor.intensity', intensity)
    thisExp.addData('Gabor.pos', Gabor.pos)
    thisExp.addData('Gabor.ori', Gabor.ori)
    thisExp.addData('Left_Cue.opacity', Left_Cue.opacity)
    thisExp.addData('Right_Cue.opacity', Right_Cue.opacity)
    thisExp.addData('Condition', trial['cue_condition'])

    if not practice:
        thisExp.addData('QP_Threshold', threshold)
        thisExp.addData('QP_Slope', slope)
        thisExp.addData('QP_Lapse', lapse_rate)

    # Response Check
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
        response = None
        if practice:
            feedback.text = "Try Again!"
            feedback.draw()
            win.flip()
            core.wait(1)
    else:
        response = 1 if (
            (key_resp.keys == '1' and trial['orientation'] == 0) or 
            (key_resp.keys == '2' and trial['orientation'] == 90)
            ) else 0
        if not practice:
            current_qp.update(stim={'intensity': intensity}, outcome={'response': response})
        if response == 1:
            if practice:
                feedback.text = "Correct!"
                feedback.draw()
                win.flip()
                core.wait(1)
        else:
            if practice:
                feedback.text = "Try Again!"
                feedback.draw()
                win.flip()
                core.wait(1)         

    thisExp.addData('Keypress',key_resp.keys)
    thisExp.addData('Accuracy', response)
    # Save response, accuracy, RT, and duration to data file and print response
    if key_resp.keys != None: 
        thisExp.addData('RT', key_resp.rt)
        thisExp.addData('Keypress Duration', key_resp.duration)
    
    print("Accuracy:", response)

    routineTimer.reset()  # Reset the routine timer for the next trial
    thisExp.nextEntry() # Advance to the next row in the data file
        
    return response


####### PRACTICE BLOCK #################################################################################################################################################################################################### 

routineTimer.reset()

repeat_count = 0
repeat_practice = True
while repeat_practice:
    repeat_count += 1
    total_correct = 0

    trial_list = get_trials(PRACTRIALS_REPETITIONS)

    pract_contrasts = PRACT_CONTRASTS.copy() 
    shuffle(pract_contrasts)

    for trial in trial_list:
        thisExp.addData('Trial', trial['Index'])
        print(f"Practice Trial {trial['Index']}:", trial)
        answer = run_trial(trial, practice=True)

        if answer != None: 
            total_correct += answer

    thisExp.addData('Practice.stopped', globalClock.getTime(format='float'))

    # Calculate accuracy on practice trials, determine whether to continue to experiment block, repeat practice, or end the experiment
    percent_correct = (total_correct / (len(trial_list))) * 100
    print(f"Percent correct: {percent_correct}")

    if percent_correct >= 75: # Continue to experiment block
        repeat_practice = False
        start_text = visual.TextStim(win, text="Great Job!\n\nAre you ready to play the real game? ", color="black", units='pix', height=40, wrapWidth = 1700)
        print("Practice block passed! Starting experiment trials.")
        start_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
    else:
        if repeat_count < 2: # Repeat practice block if less than 75% accuracy on first try
            retry_text = visual.TextStim(win, text="Let's try the practice again!", color="black", units='pix', height=40, wrapWidth=1700)
            retry_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            routineTimer.reset()
        else: # End experiment if accuracy below 75% after two attempts
            print("Participant did not pass practice trials. Ending experiment.")
            end_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            win.close()
            core.quit()

####### EXPERIMENT BLOCK #################################################################################################################################################################################################### 

thisExp.addData('Experiment.started', globalClock.getTime(format='float'))

routineTimer.reset()
no_resp_trials = []

trial_list = get_trials(TRIAL_REPETITIONS)
for trial in trial_list:
    trial['presented'] = 1 # start keeping track of how many times each trial has been presented 

for trial in trial_list:
    thisExp.addData('Trial', trial['Index'])
    thisExp.addData('Presentations', trial['presented'])
    print(f"Experiment Trial {trial['Index']}:", trial)

    answer = run_trial(trial)

    if answer is None:
        no_resp_trials.append(trial)  

    # Give break every interval
    if trial['Index'] % BREAK_INTERVAL == 0 and trial['Index'] != len(trial_list): 
        break_text = visual.TextStim(win, text="Take a quick break!", color="black", units='pix', height=40, wrapWidth=1700)
        break_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        routineTimer.reset()

# Repeat trials with no response
while len(no_resp_trials) > 0:
    routineTimer.reset()
    print(f"Re-running {len(no_resp_trials)} trials with no response...")

    remaining_trials = []

    for trial in no_resp_trials:
        trial_num = trial_list.index(trial) + 1
        thisExp.addData('Trial', trial_num)
        print(f"Re-running Trial {trial_num} (Attempt {trial['presented']}):", trial)

        answer = run_trial(trial)

        if answer is None and trial['presented'] < MAX_PRESENTATIONS:
            remaining_trials.append(trial)

    no_resp_trials = remaining_trials

####### SAVE DATA #################################################################################################################################################################################################### 

filename = thisExp.dataFileName
# these shouldn't be strictly necessary (should auto-save) but just in case
thisExp.saveAsWideText(filename + '.csv', delim='auto')
thisExp.saveAsPickle(filename)

####### END EXPERIMENT #################################################################################################################################################################################################### 

end_text.draw()
win.flip()
event.waitKeys(keyList=['space'])

if win is not None:
    # remove autodraw from all current components
    win.clearAutoDraw()
    win.flip()
thisExp.status = FINISHED
logging.flush()

thisExp.abort()  # or data files might save again on exit
if win is not None:
    win.flip()
    win.close()
# shut down eyetracker, if there is one
if deviceManager.getDevice('eyetracker') is not None:
    deviceManager.removeDevice('eyetracker')
logging.flush()
# terminate Python process
core.quit()