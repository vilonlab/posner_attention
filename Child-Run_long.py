# --- Import packages ---
from psychopy import prefs, plugins, sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, monitors
from psychopy.constants import (NOT_STARTED, STARTED, STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy.tools.monitorunittools import pix2deg
from psychopy.hardware import keyboard
import numpy as np 
import math
from numpy.random import random, randint, normal, shuffle, choice as randchoice
from string import ascii_letters, digits
import os
import sys 
import time
from questplus import QuestPlus
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


####### EXPERIMENT PARAMETERS ####################################################################################################################################################################################################

globalClock = core.Clock() # initialize global clock

EYETRACKER_OFF = False # Set to True to run the script without eyetracking
current_qp = None # Setting global variable so then we can print posteriors at the end
RESPONSE_KEYS = ['1', '2'] # 1 for left, 2 for right

# get 8 unique trial types by combining the trial variables (e.g., one trial type is: {'orientation': 0,  'gabor_position': -1, 'cue_condition': 'Neutral'})
TRIAL_TYPES = data.createFactorialTrialList({
            'orientation': [0, 90], # 0 - vertical; 90 - horizontal
            'gabor_position': [-1, 1], # -1 = Left, 1 = Right
            'cue_condition': ['Neutral', 'Valid']  
            })

# practice blocks
PTRIAL_PRESENTATIONS = 2 # how many times to present each of the 8 unique TRIAL_TYPES in each practice block
PTOTAL_TRIALS = PTRIAL_PRESENTATIONS * len(TRIAL_TYPES) # total trials in practice blocks 2 and 3
PRACT1_CONTRASTS = 1.0 # 100% contrast for all trials in practice block 1
PRACT_CONTRASTS = [0.1, 0.4, 0.7, 1.0] # hardcoded possible gabor contrast values for practice trials; keep length to a factor of 8
EXTENDED_TARGET_DUR = 1.0 # target gabor duration for practice block 2
ACCURACY_THRESHOLD = 62 # accuracy needed to pass the practice blocks 
MAX_PRACTICE_REPEATS = 2 # maximum number of times each practice block can be repeated before experiment ends

# experiment blocks
TRIAL_PRESENTATIONS =  16 # how many times to present each of the 8 unique TRIAL_TYPES throughout all experiment blocks
TOTAL_TRIALS = TRIAL_PRESENTATIONS * len(TRIAL_TYPES) # total number of experiment trials
MAX_CONSECUTIVE_TRIALS = 3 # maximum number of consecutive trials of the same cue condition (valid, neutral)
MAX_TRIAL_REPEATS = 3 # maximum number of times each trial can be presented after being aborted (includes initial presentation)

# stims (deg)
CUE_SIZE = .5
TARGET_SIZE = 2
FIXCROSS_SIZE = 0.75
ANDYFIX_SIZE = 1.5
POSITION = np.array([6.0, 0.0]) # DVA eccentricity for target and cues
GAZE_BOUNDS = 3 # if gaze shifts more than this from fixation point, trial is aborted 
SPATIAL_FREQUENCY = 5

# timing (s)
frameTolerance = 0.001  # How close to onset before 'same' frame
FIX_CROSS_DUR = 1 # duration of fixation cross at start of trial
ANDY_FIX_DUR = 1.5 # duration of andy fixation right before cue
CUE_DUR = 0.05 # cue duration
ISI = 0.05 # duration of andy fixation between cue and target 
EXP_TARGET_DUR = 0.5 # target gabor duration for experiment trials
RESPONSE_WINDOW = 1.0 # duration of andy fixation after target offset; total response window is TARGET_DUR + RESPONSE_WINDOW
FEEDBACK_DUR = 1.0 # duration of feedback presentation for practice blocks
LOSS_THRESHOLD = 0.1 # maximum amount of time sample can lose track of the eye before a trial is aborted 

####### WINDOW, DATA FILE, & EYETRACKER SETUP ####################################################################################################################################################################################################

# Collect participant ID, visit number, number of blocks and check that the inputted variables are valid
exp_name = 'ZebraFliesTask'
exp_info = {
    'Participant ID': '',
    'Visit': '',
    'Blocks':'Ex. 4,6,8'}
while True:
    dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name)
    if dlg.OK == False:
        core.quit()
        sys.exit()

    # get blocks and write edf filename
    blocks = int(exp_info['Blocks'])
    participant_id = exp_info['Participant ID']
    edf_filename = f"{participant_id}_ET"

    # check if the filename and number of blocks are valid
    allowed_char = ascii_letters + digits + '_'
    if not all([c in allowed_char for c in edf_filename]):
        raise ValueError('ERROR: Invalid EDF filename. Enter only letters, digits, or underscores.')
    elif len(edf_filename) > 8:
        raise ValueError("ERROR: Invalid EDF filename: participant ID must be ≤5 characters.")
    elif (TOTAL_TRIALS)%blocks != 0:
        raise ValueError(f"ERROR: Invalid number of blocks. Must be a factor of {TOTAL_TRIALS}.")
    else:
        break

# Calculate number of trials in each block
trials_per_block = TOTAL_TRIALS/blocks
print("Trials per block:", trials_per_block)

# Establish data output directory
time_str = time.strftime("_%m_%d_%Y_%H-%M", time.localtime())
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"{participant_id}_{exp_name}_Visit{exp_info['Visit']}{time_str}")
os.makedirs(output_folder, exist_ok=True)
filename = os.path.join(output_folder, f"{participant_id}_{exp_name}_Visit{exp_info['Visit']}") # file for psychopy task data
edf_path = os.path.join(output_folder, f"{edf_filename}.EDF") # file for eyetracker data
logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # set logging level: warnings, errors, and critical messages will be displayed in output console

# Window setup for EIZO monitor
view_dist_cm = 60
screen_w_cm = 51.84
screen_w_px = 1920
Eizo = monitors.Monitor('Eizo', width = screen_w_cm, distance = view_dist_cm)
Eizo.setSizePix([1920, 1200])
win = visual.Window(fullscr=True, color=[0,0,0],
            size=Eizo.getSizePix(), screen=1,
            winType='pyglet', allowStencil=False,
            monitor=Eizo, colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=False,
            units='deg', checkTiming=False)
 
# Get the screen resolution used by PsychoPy
scn_width, scn_height = win.size # in retina pixels

# Calculate host PC pixel conversions
host_x = int(scn_width//2)
host_y = int(scn_height//2)
px_per_cm = screen_w_px / screen_w_cm
px_per_dva = px_per_cm * (2 * view_dist_cm * math.tan(math.radians(0.5)))

# Save frame rate to data file
exp_info['frameRate'] = win.getActualFrameRate() 

# Create an experiment handler
thisExp = data.ExperimentHandler(
    name=exp_name, version='',
    extraInfo=exp_info, runtimeInfo=None,
    originPath=os.path.abspath(__file__),
    savePickle=True, saveWideText=True,
    dataFileName=filename, 
    sortColumns='time') 

# -------------- Connect to the EyeLink Host PC; based on sample scripts from SR Research
if EYETRACKER_OFF:
    el_tracker = pylink.EyeLink(None) # Need this line so code will run when not using eyetracking
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()
        
# Open the EDF data file on the Host PC
try:
    el_tracker.openDataFile(edf_filename)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add header text to EDF file for data viewing
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

# Get eyetracker version/model; EyeLink 1000 Plus is version 5
eyelink_ver = 0  # Set to 0 so code will run when not using eyetracking
if not EYETRACKER_OFF:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])

# Set what eye events to save in the EDF file and make available over the link, include everything by default
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
# Set what sample data to save in the EDF data file and to make available over the link
if eyelink_ver > 3:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
else: # For when running without eyetracking
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Choose a calibration type (HV = horizontal/vertical) number is how many points on the screen
el_tracker.sendCommand("calibration_type = HV5")

# Pass the display pixel coordinates (left, top, right, bottom) to the tracker
el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendCommand(el_coords)

# Write a DISPLAY_COORDS message to the EDF file
# Data Viewer needs this piece of info for proper visualization
dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendMessage(dv_coords)

# Configure a graphics environment (genv) for tracker calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)

# Set visuals for calibration routine
foreground_color = (-1, -1, -1)
background_color = win.color
genv.setCalibrationColors(foreground_color, background_color)
genv.setTargetType('picture')
genv.setPictureTarget(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'andy_fixation.png')) 

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('sounds/boing.wav', '', '')

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)
logging.info(f"Graphics environment set up: {genv}")

####### QUESTPLUS INITIALIZATION ####################################################################################################################################################################################################

stim_domain = {'intensity': np.arange(0.01, 1, 0.01)}
param_domain = {
    'threshold': np.arange(0.01, 1, 0.01),
    'slope': 3.5,
    'lower_asymptote': 0.5, # Equal to chance
    'lapse_rate': np.arange(0, 0.05, 0.01) # Test 0:0.05 for adults, Consider 0:0.10 for children
}
outcome_domain = {'response': [1,0]}  # I'm going to flip this, to see if it fixes the way I intuitively think the algorithm should work; TDW 2025-01-22

# *TWO* QuestPlus staircases - one for each condition
qp_valid = QuestPlus(
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

####### INITIALIZE EXPERIMENT TRIAL COMPONENTS #################################################################################################################################################################################################### 

kb = keyboard.Keyboard()

fix_cross = visual.ShapeStim(
    win=win, name='fix_cross', vertices='cross',units='deg', 
    size=(FIXCROSS_SIZE, FIXCROSS_SIZE),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=0.0, interpolate=True)
andy_fix = visual.ImageStim(win=win,
    image = "images/andy_fixation.png",
    name='andy_fix', units='deg', 
    mask=None, ori=0, pos=(0, 0), 
    size=(ANDYFIX_SIZE, ANDYFIX_SIZE),
    colorSpace='rgb')
left_cue = visual.ShapeStim(
    win=win, name='left_cue',units='deg', 
    size=(CUE_SIZE, CUE_SIZE), vertices='circle',
    ori=0.0, pos=(-POSITION[0], POSITION[1]), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=-1.0, interpolate=True)
right_cue = visual.ShapeStim(
    win=win, name='right_cue',units='deg', 
    size=(CUE_SIZE, CUE_SIZE), vertices='circle',
    ori=0.0, pos=(POSITION[0], POSITION[1]), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=-1.0, interpolate=True)
gabor = visual.GratingStim(
    win=win, name='gabor',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=0.0, pos=(POSITION[0],POSITION[1]), size=(TARGET_SIZE, TARGET_SIZE), sf=(SPATIAL_FREQUENCY), phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True)
feedback_image = visual.ImageStim(win=win,
    name='feedback_image', units='deg', 
    image='sin', mask=None,
    ori=0, pos=(0, 2.5), size=(5, 5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
feedback_text = visual.TextStim(win=win, name='feedback_text',
    text="", font='Arial',
    units='deg', pos=(0, -2.5), draggable=False, height=1.2, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',depth=0.0)
happy_sound = sound.Sound('sounds/happy_ribbit.wav')
sad_sound = sound.Sound('sounds/sad_ribbit.wav')
gaze_feedback = visual.TextStim(win=win, name='gaze_feedback',
    text="", font='Arial',
    units='deg', pos=(0, -2.5), draggable=False, height=1.2, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',depth=0.0)
prac_outcome_text = visual.TextStim(win=win, name='prac_outcome_text',
    text="", font ='Arial', color= 'black',
    units='deg', pos=(0, 0), draggable=False, height=1.2, wrapWidth=1700, ori=0)
break_text = visual.TextStim(win=win, name='break_text',
    text="Great job!\nLet's take a quick break!", font ='Arial', color= 'black',
    units='deg', pos=(0, 0), draggable=False, height=1.2, wrapWidth=1700, ori=0)
end_text = visual.TextStim(win=win, name='end_text',
    text="You finished the game!", font='Arial', units = 'deg',
    pos=(0, 0), draggable=False, height=1.5, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',depth=0.0)

####### FUNCTIONS #################################################################################################################################################################################################### 

def abort_trial(trial_index = 0, practice = False, block_num = 0):
    """ Ends trial and clears the eyetracker """
    
    # Stop recording
    el_tracker = pylink.getEYELINK()
#    pylink.pumpDelay(100) # add 100 ms to catch final trial events
    el_tracker.stopRecording()

    # Clear the psychopy window
    if win is not None:
        win.clearAutoDraw()
        win.flip()
    
    # Send a message to clear the Data Viewer screen
    el_tracker.sendMessage('!V CLEAR 116 116 116')
    
    if practice and block_num != 0:
        feedback_text.text = "Oops! You looked at the fly!"
        feedback_image.setImage("images/eye.png")
        feedback_image.draw()
        feedback_text.draw()
        sad_sound.play()
        win.flip()
        core.wait(FEEDBACK_DUR)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
    
    print(f'Trial {trial_index} aborted.')
    thisExp.addData('trial.aborted', 'aborted')
    thisExp.nextEntry()

    return None
    
def terminate_task():
    """ Disconnects eyetracker, closes psychopy, and saves all data files """
    
    # Save task data
    thisExp.nextEntry()
    thisExp.addData('experiment.end', globalClock.getTime(format='float'))
    thisExp.saveAsWideText(thisExp.dataFileName + '.csv', delim='auto')
    thisExp.saveAsPickle(thisExp.dataFileName)
    logging.flush()
    
    # Clear the psychopy window
    if win is not None:
        win.clearAutoDraw()
        win.flip()
    
    # Mark experiment as finished
    thisExp.status = FINISHED
    print("Experiment ended.")
    thisExp.abort()

    if not EYETRACKER_OFF:
        # Disconnect eyetracker
        el_tracker = pylink.getEYELINK()
        if el_tracker.isConnected():
            # Abort trial
            error = el_tracker.isRecording()
            if error == pylink.TRIAL_OK:
                abort_trial()
            # Put tracker in Offline mode
            el_tracker.setOfflineMode()
            # Clear the Host PC screen and wait for 500 ms
            el_tracker.sendCommand('clear_screen 0')
            pylink.msecDelay(500)
            # Close the edf data file on the Host
            el_tracker.closeDataFile()
            # Print a file transfer message
            print('EDF data is transferring from EyeLink Host PC...')
            # Download the EDF data file from the Host PC to a local data folder
            try:
                el_tracker.receiveDataFile(edf_filename, edf_path)
                print(f"EDF file saved to: {edf_path}")
            except RuntimeError as error:
                print('ERROR downloading EDF file:', error)
            # Close the link to the tracker
            el_tracker.close()
            
    # quit psychopy
    win.close()
    core.quit()
    sys.exit()

def get_cue_opacity(cue_condition, gabor_position):
    """ Returns the opacities of the left and right cues depending on the trial's cue condition. """
    
    if cue_condition == 'Neutral': # show both cues
        return 1.0, 1.0 
    elif cue_condition == 'Valid': # show cue in same position as target gabor
        if gabor_position == -1:
            return 1.0, 0.0 
        else:
            return 0.0, 1.0

def consecutive_check(trial_list):
    """ Checks the trial list to ensure that there are no more than MAX_CONSECUTIVE_TRIALS consecutive trials with the same cue condition """
    consecutive_count = 1
    if len(trial_list) == 0:
        return False
    for i in range(1, len(trial_list)):
        if trial_list[i]['cue_condition'] == trial_list[i - 1]['cue_condition']:
            consecutive_count += 1
            if consecutive_count > MAX_CONSECUTIVE_TRIALS:
                return False  # Invalid trial list
        else:
            consecutive_count = 1
    return True  # Valid trial list

# Get the full list of trials created by the handler
def create_trial_list(block_type):
    """ Create the trial list for the block """
    if block_type == 'practice1':
        reps = 1
    elif block_type == 'practice':
        reps = PTRIAL_PRESENTATIONS
    elif block_type == 'experiment':
        reps = TRIAL_PRESENTATIONS
    trial_list = []
    
    while consecutive_check(trial_list) == False:
        handler = data.TrialHandler(nReps=reps, method='random', 
            extraInfo=exp_info, originPath=-1,
            trialList=TRIAL_TYPES,
            seed=None, name='handler')
        trial_sequence = handler.sequenceIndices 
        trial_indices = trial_sequence.T.flatten().tolist()
        trial_list = [dict(handler.trialList[i]) for i in trial_indices] 
        for i, trial in enumerate(trial_list, start=1):
            trial['index'] = i
            trial['presented'] = 0
            
    return trial_list

def draw_comp(comp, t, tThisFlipGlobal, frameN):
    """ Draw visual components and save onset to the data file """
    comp.frameNStart = frameN
    comp.tStart = t
    comp.tStartRefresh = tThisFlipGlobal
    win.timeOnFlip(comp, 'tStartRefresh')
    comp.status = STARTED
    if not isinstance(comp, keyboard.Keyboard):  # kb is handled differently
        thisExp.timestampOnFlip(win, f'{comp.name}.start')
        comp.setAutoDraw(True)
    else:
        thisExp.timestampOnFlip(win, 'kb.start')

def erase_comp(comp, t, tThisFlipGlobal, frameN):
    """ Erase visual components and save offset to the data file """
    comp.tStop = t
    comp.tStopRefresh = tThisFlipGlobal
    comp.frameNStop = frameN
    comp.status = FINISHED
    if not isinstance(comp, keyboard.Keyboard):
        thisExp.timestampOnFlip(win, f'{comp.name}.end')
        comp.setAutoDraw(False)
    else:
        thisExp.timestampOnFlip(win, 'kb.end')

def get_eye_used(el_tracker):
    """ Gets eye used. Returns 0 for left, 1 for right, None if eye data cannot be collected.t"""
    if EYETRACKER_OFF:
        return 0 # For when running without eyetracking
        
    if el_tracker is not None and el_tracker.isConnected():
        eye = el_tracker.eyeAvailable()
        if eye in [0, 1]:
            el_tracker.sendMessage(f"EYE_USED {eye} {'RIGHT' if eye == 1 else 'LEFT'}")
            return eye
        elif eye == 2: # binocular vision defaults to left eye
            el_tracker.sendMessage("EYE_USED 0 LEFT")
            return 0
            
    print("ERROR: EyeLink not connected or invalid eye")
    return None
    
def is_gaze_within_bounds(el_tracker, eye_used, sampleTimeList, loss_clock):
    """ Check that gaze is within GAZE_BOUNDS. If there is no sample, wait loss_threshold seconds before returning False """
    while True:
        sample = el_tracker.getNewestSample()
        
        stime = sample.getTime() * 1000
        sampleTimeList.append(stime)
        
        if eye_used == 0 and sample.isLeftSample():
            eye_data = sample.getLeftEye()
        elif eye_used == 1 and sample.isRightSample():
            eye_data = sample.getRightEye()
        else:
            if loss_clock.getTime() > LOSS_THRESHOLD:
                print("Loss of sample for longer than 100ms.")
                thisExp.addData('Loss clock', loss_clock)
                return False
            continue
        
        gaze = eye_data.getGaze()
        pupil = eye_data.getPupilSize()
        
        if pupil <= 0:
            print("Blink detected.")
            return True
            
        loss_clock.reset()
        
        dx = gaze[0] - scn_width/2
        dy = scn_height/2 - gaze[1]
        pixels = np.array([dx, dy])
        dx_deg, dy_deg = pix2deg(pixels, monitor=Eizo)
        
        return abs(dx_deg) <= GAZE_BOUNDS and abs(dy_deg) <= GAZE_BOUNDS

def show_instructions(block_num=None):
    """Display instructions based on block. """
    
    thisExp.addData('instructions.start', globalClock.getTime(format='float'))
    instruct_text = visual.TextStim(win=win, name='instruct_text',
    text="", font='Arial', units='deg', pos=(0, 0), draggable=False, 
    height=1.2, wrapWidth=1700, ori=0, color='black', colorSpace='rgb', 
    opacity=1,languageStyle='LTR',depth=0.0)
    
    gabor_inst1 = visual.GratingStim(
        win=win, name='gabor_inst1',units='deg', 
        tex='sin', mask='gauss', anchor='center',
        ori=90, pos=(5,0), size=(4,4), sf=(SPATIAL_FREQUENCY), phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=1.0, contrast=0.5, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-2.0)
    gabor_inst2 = visual.GratingStim(
        win=win, name='gabor_inst2',units='deg', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=(-5,0), size=(4,4), sf=(SPATIAL_FREQUENCY), phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=1.0, contrast=0.5, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-2.0)
        
    if block_num == 0:
        instruct_text.text = '''Zebra flies are really shy!'''
        instruct_text.draw()
        win.flip()
    elif block_num in (1, 2, 3):
        instruct_text.text = f'''***PRACTICE LEVEL {block_num}***\n\n
        Your job is to tell Andy which way the zebra flies are going!\n\n\n\n\n\n\n\n
        Press the left button if the zebra flies will move up and down, 
        and the right button if the zebra flies will move side to side.'''
        instruct_text.draw()
        andy_fix.draw()
        gabor_inst1.draw()
        gabor_inst2.draw()
        win.flip()
    else:
        instruct_text.text = '''Your job is to tell Andy which way the zebra flies are going!\n\n\n\n\n\n\n\n
        Press the left button if the zebra flies will move up and down, 
        and the right button if the zebra flies will move side to side.'''
        instruct_text.draw()
        andy_fix.draw()
        gabor_inst1.draw()
        gabor_inst2.draw()
        win.flip()

    keys = event.waitKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        terminate_task()
    elif 'space' in keys:
        thisExp.addData('instructions.end', globalClock.getTime(format='float'))
        return

def run_trial(trial, practice = False, practice_contrasts = None, block_num = None):
    """ Run one trial. Returns response: 1- Correct; 0 - Incorrent; None - No response or trial aborted """
    
    # Reset variables
    t = 0
    frameN = -1
    routineTimer.reset()
    continueRoutine = True
    allKeys = []
    components = [fix_cross, andy_fix, left_cue, right_cue, gabor, kb]
    eye_used = None
    for comp in components:
        comp.tStart = None
        comp.tStop = None
        comp.tStartRefresh = None
        comp.tStopRefresh = None
        if hasattr(comp, 'status'):
            comp.status = NOT_STARTED
    
    # Mark start of trial
    trial_index = trial['index']
    trial['presented'] += 1
    thisExp.addData('trial.start', globalClock.getTime(format='float'))
    thisExp.addData('trial', trial_index)
    thisExp.addData('presentations', trial['presented'])
    
    # Set cue and target parameters 
    left_cue.opacity, right_cue.opacity = get_cue_opacity(trial['cue_condition'], trial['gabor_position'])
    gabor.pos = np.array([POSITION[0] * trial['gabor_position'], POSITION[1]])
    gabor.ori = trial['orientation']

    # Set practice-specific variables
    if practice: 
        intensity = practice_contrasts.pop(0)
        gabor.contrast = intensity
        
        if block_num == 0:
            TARGET_DUR = None # target on screen for unlimited amount of time
            thisExp.addData('block','bio')
        if block_num == 1:
            TARGET_DUR = None # target on screen for unlimited amount of time
            thisExp.addData('block','pract1')
        elif block_num == 2:
            TARGET_DUR = EXTENDED_TARGET_DUR
            thisExp.addData('block','pract2')
        else:
            TARGET_DUR = EXP_TARGET_DUR
            thisExp.addData('block','pract3')
            
    # Set QP algorithm logic for experiment trials
    else:
        TARGET_DUR = EXP_TARGET_DUR
        thisExp.addData('block','exp')
        global current_qp
        if trial['cue_condition'] == 'Valid':
            current_qp = qp_valid
        elif trial['cue_condition'] == 'Neutral':
            current_qp = qp_neutral
        
        # Save threshold, slope, and lapse rate from QP to data file
        threshold = current_qp.param_estimate['threshold']
        slope = current_qp.param_estimate['slope']
        lapse_rate = current_qp.param_estimate['lapse_rate']
        
        # Get next intensity from current staircase
        next_stim = current_qp.next_stim
        intensity = next_stim['intensity']
        
        # Update gabor contrast
        gabor.contrast = intensity
        
    # Gaze check starting 100ms before cue
    GAZE_CHECK = [FIX_CROSS_DUR+ANDY_FIX_DUR-0.1, None]
    if TARGET_DUR is not None:
        TRIAL_DUR = FIX_CROSS_DUR+ANDY_FIX_DUR+CUE_DUR+ISI+TARGET_DUR+RESPONSE_WINDOW
    
    # -------------------- Eyetracker Setup ---------------------------------
    # Esure tracker is ready to receive commands
    el_tracker = pylink.getEYELINK()
    el_tracker.setOfflineMode()
    el_tracker.sendCommand('clear_screen 0')
 
    # draw cross at fixation point on the host pc; params: x y size (pix)
    el_tracker.sendCommand(f'draw_cross {host_x} {host_y} 10')
    
    # draw box of gaze boundary; params: x1 y1 x2 y2 color <- coordinates are for opposite points on the box
    half_gaze_bound = min(200, max(1, int(round(GAZE_BOUNDS * px_per_dva / 2)))) 
    el_tracker.sendCommand(f'draw_box {host_x-half_gaze_bound} {host_y-half_gaze_bound} '
        f'{host_x+half_gaze_bound} {host_y+half_gaze_bound} 1')
        
    #draw cross at cue/target locations
    dx = int(round(POSITION[0] * px_per_dva))
    el_tracker.sendCommand(f'draw_cross {host_x-dx} {host_y} 10')
    el_tracker.sendCommand(f'draw_cross {host_x+dx} {host_y} 10')
    
    # Print trial number on eyelink host monitor and output console
    if practice:
        print(f"Running practice trial {trial_index}:", trial)
        status_msg = 'PRACTICE TRIAL %d' % trial_index
        el_tracker.sendMessage('PRACTICE_TRIALID %d' % trial_index)
    else:
        print(f"Running experiment trial {trial_index}:", trial)
        status_msg = 'TRIAL number %d' % trial_index
        el_tracker.sendMessage('TRIALID %d' % trial_index)

    # Send status message to host PC
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)

    # put tracker in idle/offline mode before recording
    el_tracker.setOfflineMode()
    
    # Start recording
    try:
        el_tracker.startRecording(1, 1, 1, 1) # arguments: sample_to_file, events_to_file, sample_over_link, event_over_link (1-yes, 0-no)
    except RuntimeError as error:
        print("ERROR:", error)
        return abort_trial(trial_index, practice, block_num)
    
    # Allocate time for the tracker to cache some samples
    pylink.pumpDelay(100) 
    
    # Get eye used. When using eyetracking, abort trial if no eye information can be collected.
    eye_used = get_eye_used(el_tracker)
    if eye_used is None and not EYETRACKER_OFF:
        print(f"Could not get eye used on trial {trial_index}.")
        return abort_trial(trial_index, practice, block_num)
        
    # Abort trial if tracker is no longer recording
    error = el_tracker.isRecording()
    if error is not pylink.TRIAL_OK:
        el_tracker.sendMessage('tracker_disconnected')
        print("Tracker disconnected - aborting trial.")
        return abort_trial(trial_index, practice, block_num)
    # ------------------------------------------------------------------------

    sampleTimeList = list()
    loss_clock = core.Clock()
    
    # Start trial while loop
    while continueRoutine: 
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1
        
        # Draw fixation cross at start of trial for its set duration
        if fix_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            draw_comp(fix_cross, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('fixation_started')
            
        # Draw Andy fixation to signal start of trial; stays on until trial ends
        if andy_fix.status==NOT_STARTED and tThisFlip >= FIX_CROSS_DUR -frameTolerance:
            erase_comp(fix_cross, t, tThisFlipGlobal, frameN)
            draw_comp(andy_fix, t, tThisFlipGlobal, frameN)
            
        # Gaze tracking until end of trial
        if not EYETRACKER_OFF and GAZE_CHECK[0]-frameTolerance <= tThisFlip <= float('inf'):
            if not is_gaze_within_bounds(el_tracker, eye_used, sampleTimeList, loss_clock = loss_clock):
                return abort_trial(trial_index, practice, block_num)
                
        # Draw the left and right cues (onset and offset is same for both)
        if left_cue.status == NOT_STARTED and tThisFlip >= FIX_CROSS_DUR+ANDY_FIX_DUR-frameTolerance:
            draw_comp(left_cue, t, tThisFlipGlobal, frameN)
            draw_comp(right_cue, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('cue_started')
        if left_cue.status == STARTED and tThisFlipGlobal > left_cue.tStartRefresh + CUE_DUR-frameTolerance:
            erase_comp(left_cue, t, tThisFlipGlobal, frameN)
            erase_comp(right_cue, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('cue_stopped')
            
        # Draw target
        if gabor.status == NOT_STARTED and tThisFlip >= (FIX_CROSS_DUR+ANDY_FIX_DUR+CUE_DUR+ISI)-frameTolerance:
            draw_comp(gabor, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('target_started')
            
        # Only erase target if target_dur is defined (i.e., target is on screen indefinitely in practice block 1)
        if TARGET_DUR is not None:
            if gabor.status == STARTED and tThisFlipGlobal > gabor.tStartRefresh + TARGET_DUR -frameTolerance:
                erase_comp(gabor, t, tThisFlipGlobal, frameN)
                el_tracker.sendMessage('target_stopped')
        
        # Start checking for key response on target onset
        if kb.status == NOT_STARTED and tThisFlip >= (FIX_CROSS_DUR+ANDY_FIX_DUR+CUE_DUR+ISI)-frameTolerance:
            draw_comp(kb, t, tThisFlipGlobal, frameN)
            win.callOnFlip(kb.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(kb.clearEvents, eventType='keyboard')
            
        if kb.status == STARTED: 
            key_name = None # None unless response is made
            rt = None # None unless response is made
            key = kb.getKeys(keyList=RESPONSE_KEYS, waitRelease=False)
            allKeys.extend(key)
            
            if allKeys: # end trial loop when a key is pressed
                last_key = allKeys[-1]
                key_name = last_key.name  # get just the last key pressed
                rt = last_key.rt
                continueRoutine = False 
                break
                
            if not (block_num == 1 or block_num == 0): # if not biofeedback or practice block 1, end trial after trial_dur if no key is pressed
                if tThisFlip >= TRIAL_DUR - frameTolerance: 
                    continueRoutine = False
                    break
            
        if kb.getKeys(keyList=["escape"]):
            terminate_task()
        
        if continueRoutine:
            win.flip()
            
    for comp in components:
        if hasattr(comp, 'setAutoDraw'):
            comp.setAutoDraw(False)
            
    # Add trial end time to data file
    thisExp.addData('trial.end', globalClock.getTime(format='float'))
    
    # Response check and feedback 
    if key_name is None:  # No response was made, do not update qp, show feedback in practice blocks
        response = None
        if practice and block_num != 0:
            feedback_text.text = "Remember to press a button!"
            feedback_image.setImage("images/x_mark.png")
            feedback_image.draw()
            feedback_text.draw()
            sad_sound.play()
            win.flip()
            core.wait(FEEDBACK_DUR)
    else: # response was made, check accuracy, update qp, show feedback in practice blocks
        response = 1 if (
            (key_name == '1' and trial['orientation'] == 0) or 
            (key_name == '2' and trial['orientation'] == 90)
            ) else 0
        if not practice:
            current_qp.update(stim={'intensity': intensity}, outcome={'response': response})
        elif practice:
            if response == 1:
                feedback_text.text = "Correct!"
                feedback_image.setImage("images/check_mark.png")
                feedback_image.draw()
                feedback_text.draw()
                happy_sound.play()
                win.flip()
                core.wait(FEEDBACK_DUR)
            elif response == 0:
                feedback_text.text = "Try Again!"
                feedback_image.setImage("images/x_mark.png")
                feedback_image.draw()    
                feedback_text.draw()
                sad_sound.play()
                win.flip()
                core.wait(FEEDBACK_DUR)
    print("Response:", response)
    
    # Add trial data to the data file
    thisExp.addData('gabor.intensity', intensity)
    thisExp.addData('gabor.pos', 'L' if trial['gabor_position'] == -1 else 'R')
    thisExp.addData('gabor.ori', gabor.ori)
    thisExp.addData('left_cue.opacity', left_cue.opacity)
    thisExp.addData('right_cue.opacity', right_cue.opacity)
    thisExp.addData('condition', trial['cue_condition'])
    thisExp.addData('keypress', key_name)
    thisExp.addData('accuracy', response)
    if key_name != None: 
        thisExp.addData('rt', rt)
    if not practice:
        thisExp.addData('qp_threshold', threshold)
        thisExp.addData('qp_slope', slope)
        thisExp.addData('qp_lapse', lapse_rate)
        print(f"Next Intensity: {intensity}")
    
    # Send trial data to EDF file
    el_tracker.sendMessage('!V TRIAL_VAR condition %s' % trial['cue_condition'])
    try:
        el_tracker.sendMessage('!V TRIAL_VAR RT %d' % int(rt))
    except (TypeError, ValueError):
        el_tracker.sendMessage('!V TRIAL_VAR RT -1') # If no response, RT is set to -1 in eyetracker data
    
    el_tracker.sendMessage('!V CLEAR 128 128 128')
    
    # Stop recording between trials to decrease size of output file
    pylink.pumpDelay(100) # add 100 msec to catch final events before stopping
    el_tracker.stopRecording()
    
    # Send trial result message to mark the end of the trial
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    thisExp.nextEntry() # Advance to the next row in the data file
        
    return response
    
def run_biofeedback():
    """Run biofeedback trial. Repeats if 'r' is pressed, ends when 'space' is pressed Can be repeated up to 12 times."""
    
    thisExp.addData(f'biofeedback.start', globalClock.getTime(format='float'))
    trial_list = create_trial_list('practice')
    practice_contrasts = PRACT_CONTRASTS * (PTOTAL_TRIALS//len(PRACT_CONTRASTS))
    shuffle(practice_contrasts)
    trial_index = 0
    
    while True:
        show_instructions(0)
        
        response = run_trial(trial_list[trial_index], practice = True, practice_contrasts=practice_contrasts, block_num = 0)
        
        prac_outcome_text.text = "The zebra fly flew away!"
        prac_outcome_text.draw()
        win.flip()
        
        keys = event.waitKeys(keyList=['r', 'space', 'escape'])
        
        for key in keys:
            if key == 'escape':
                terminate_task()
            elif key == 'r': # show a different trial if r is pressed
                trial_index +=1
                break
            elif key == 'space':
                thisExp.addData(f'biofeedback.end', globalClock.getTime(format='float'))
                return
    
def run_practice_block(block_num):
    """ Run all trials practice block. Must surpass accuracy threshold to move on. Experiment will terminate if accuracy threshold
        is not met within two tries. """
        
    show_instructions(block_num)
    
    accuracy = 0
    repeat_count = 0
    
    if block_num == 1:
        thisExp.addData(f'practice{block_num}.start', globalClock.getTime(format='float'))
        correct_count = 0
        repeat_count +=1 
        
        trial_list = create_trial_list('practice1')
        
        practice_contrasts = [PRACT1_CONTRASTS]*len(TRIAL_TYPES)
        
        for trial in trial_list:
            response = run_trial(trial, practice = True, practice_contrasts = practice_contrasts, block_num = block_num)
            if response is not None:
                correct_count += response
            event.waitKeys(keyList=['space'])
            
        accuracy = (correct_count/len(trial_list))*100
        print(f"Practice block {block_num}, try {repeat_count}, accuracy: {accuracy}")
            
        if accuracy >= ACCURACY_THRESHOLD:
            prac_outcome_text.text = "Great Job!\n\nReady for the next level?"
            print(f"Practice block {block_num} passed!")
            prac_outcome_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            thisExp.addData(f'practice{block_num}.end', globalClock.getTime(format='float'))
            return
            
        else:
            if repeat_count < 2:
                prac_outcome_text.text = "Let's try that again!"
                prac_outcome_text.draw()
                win.flip()
                event.waitKeys(keyList=['space'])
                show_instructions(block_num)
                
            elif repeat_count == 2:
                print(f"Practice block {block_num} failed twice. Ending experiment.")
                end_text.draw()
                win.flip()
                event.waitKeys(keyList=['space'])
                terminate_task()
            
        thisExp.addData(f'practice{block_num}.end', globalClock.getTime(format='float'))
    
    elif block_num != 1:
        while accuracy <= ACCURACY_THRESHOLD and repeat_count < MAX_PRACTICE_REPEATS:
            thisExp.addData(f'practice{block_num}.start', globalClock.getTime(format='float'))
            correct_count = 0
            repeat_count +=1
            
            trial_list = create_trial_list('practice')
            
            practice_contrasts = PRACT_CONTRASTS * (PTOTAL_TRIALS//len(PRACT_CONTRASTS))
            shuffle(practice_contrasts)
            
            for trial in trial_list:
                response = run_trial(trial, practice = True, practice_contrasts =practice_contrasts, block_num = block_num)
                if response is not None:
                    correct_count += response
                    
            accuracy = (correct_count/PTOTAL_TRIALS)*100
            print(f"Practice block {block_num}, try {repeat_count}, accuracy: {accuracy}")
            
            if accuracy >= ACCURACY_THRESHOLD:
                prac_outcome_text.text = "Great Job!\n\nReady for the next level?"
                print(f"Practice block {block_num} passed!")
                prac_outcome_text.draw()
                win.flip()
                event.waitKeys(keyList=['space'])
                thisExp.addData(f'practice{block_num}.end', globalClock.getTime(format='float'))
                return
                
            else:
                if repeat_count < 2:
                    prac_outcome_text.text = "Let's try that again!"
                    prac_outcome_text.draw()
                    win.flip()
                    event.waitKeys(keyList=['space'])
                    show_instructions(block_num)
                    
                elif repeat_count == 2:
                    print(f"Practice block {block_num} failed twice. Ending experiment.")
                    end_text.draw()
                    win.flip()
                    event.waitKeys(keyList=['space'])
                    terminate_task()
                
            thisExp.addData(f'practice{block_num}.end', globalClock.getTime(format='float'))
        
    thisExp.nextEntry()

####### WELCOME SCREEN AND CALIBRATION  #################################################################################################################################################################################################### 

# Set clocks and start experiment
routineTimer = core.Clock()
logging.setDefaultClock(globalClock)
win.flip()

thisExp.addData('welcome.start', globalClock.getTime(format='float'))
thisExp.status = STARTED

# Window needs to be flipped before trying to run calibration or else code won't run
welcome_text = visual.TextStim(win=win, name='welcome_text',
    text='''Welcome to the Zebra Flies Game!''',
    font='Arial', units='deg', 
    pos=(0, 0), draggable=False, height=1.5, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',depth=0.0)
welcome_text.draw()
win.flip()

# Calibrate eyetracker if on
if not EYETRACKER_OFF:
    thisExp.addData('calibration.start', globalClock.getTime(format='float'))
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()
    thisExp.addData('calibration.end', globalClock.getTime(format='float'))
else:
    keys = event.waitKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        terminate_task()
    elif 'space' in keys:
        thisExp.addData('welcome.end', globalClock.getTime(format='float'))

####### INSTRUCTIONS  #################################################################################################################################################################################################### 

# Andy screen
andy_text = visual.TextStim(win=win, text="This is Andy the Frog!", font='Arial', units='deg', pos=(0, 6), height=1.2, wrapWidth=1700, 
    color='black', colorSpace='rgb')
andy_text.draw()
andy_fix.draw()
win.flip()
keys = event.waitKeys(keyList=['space', 'escape'])
if 'escape' in keys:
    terminate_task()
    
# Gabors screen
gabors_text = visual.TextStim(win=win, text="Andy loves to eat zebra flies like these!\n\n\n\n\n\n\n\n\n\n\n", 
    font='Arial', units='deg', pos=(0, 0), height=1.2, wrapWidth=1700, 
    color='black', colorSpace='rgb')
zebraflies_img = visual.ImageStim(win=win,
    image = "Images/zebraflies.png",
    name='zebraflies_img', units='deg', 
    mask=None, ori=0, pos=(0, -3), 
    size = (30,18.75), colorSpace='rgb')

gabors_text.draw()
zebraflies_img.draw()
win.flip()
keys = event.waitKeys(keyList=['space', 'escape'])
if 'escape' in keys:
    terminate_task()

####### PRACTICE BLOCKS #################################################################################################################################################################################################### 

run_biofeedback() # one trial for pts to look at gabor patch, can be repeated by pressing 'r'
run_practice_block(1) # unlimited amount of time for pt to answer, trial ends with experimenter pressing space
run_practice_block(2) # target presented for extended time
run_practice_block(3) # exactly like experiment trials
    
####### EXPERIMENT BLOCKS #################################################################################################################################################################################################### 

# Instruction text screen before experiment trials
show_instructions()

# Reset variables and generate the trial list
no_resp_trials = []
trial_list = create_trial_list('experiment')

for trial in trial_list:
    response = run_trial(trial, practice = False, practice_contrasts = None, block_num = None)
    if response is None:
        no_resp_trials.append(trial)  

    # At every break interval, do a drift check to recalibrate if necessary
    if (trial['index'] % trials_per_block == 0 and trial['index'] != TOTAL_TRIALS) or (trial['index'] == TOTAL_TRIALS and len(no_resp_trials)>1): 
        
        break_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        print('Number of trials to be repeated:', len(no_resp_trials)) # print total number of trials with no response so far
        
        # the doDriftCorrect() function requires target position in integers
        # the last two arguments:
        # draw_target (1-default, 0-draw the target then call doDriftCorrect)
        # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
        while not EYETRACKER_OFF:
            # terminate the task if no longer connected to the tracker
            if (not el_tracker.isConnected()) or el_tracker.breakPressed():
                terminate_task()
                
            # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                error = el_tracker.doDriftCorrect(int(scn_width/2.0),int(scn_height/2.0), 1, 1)
                # break following a success drift-check
                if error is not pylink.ESC_KEY:
                    break
            except:
                pass

# Repeat trials with no response
trial_count = TOTAL_TRIALS
while len(no_resp_trials) > 0:
    print(f"Re-running {len(no_resp_trials)} trials with no response...")

    remaining_trials = []

    for trial in no_resp_trials:
        trial_num = trial_list.index(trial) + 1
        thisExp.addData('trial', trial_num)
        trial_count += 1
        
        # At every break interval, do a drift check to recalibrate if necessary
        if trial_count % trials_per_block == 0: 
            
            print('Number of trials to be repeated:', len(no_resp_trials)) # print total number of trials with no response so far
            break_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            
            # the doDriftCorrect() function requires target position in integers
            # the last two arguments:
            # draw_target (1-default, 0-draw the target then call doDriftCorrect)
            # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
            while not EYETRACKER_OFF:
                # terminate the task if no longer connected to the tracker
                if (not el_tracker.isConnected()) or el_tracker.breakPressed():
                    terminate_task()
                    
                # drift-check and re-do camera setup if ESCAPE is pressed
                try:
                    error = el_tracker.doDriftCorrect(int(scn_width/2.0),int(scn_height/2.0), 1, 1)
                    # break following a success drift-check
                    if error is not pylink.ESC_KEY:
                        break
                except:
                    pass

        response = run_trial(trial, practice = False, practice_contrasts = None, block_num = None)

        if response is None and trial['presented'] < MAX_TRIAL_REPEATS:
            remaining_trials.append(trial)

    no_resp_trials = remaining_trials

####### END EXPERIMENT #################################################################################################################################################################################################### 

# End screen of Andy jumping up and down
amplitude = 2.0
speed = 0.8 
andy_fix.size = (7,7)
end_text.pos = (0,-6)

while True:
    t = globalClock.getTime()
    jump_y = abs(np.sin(2*np.pi*speed*t))*amplitude
    andy_fix.pos = (0, jump_y)
    
    andy_fix.draw()
    end_text.draw()
    win.flip()
    
    keys = event.getKeys(keyList=['escape'])
    if 'escape' in keys:
        terminate_task()
        break