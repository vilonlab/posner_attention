# --- Import packages ---
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, monitors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)
from psychopy.tools.monitorunittools import pix2deg
import numpy as np 
from numpy.random import random, randint, normal, shuffle, choice as randchoice
from string import ascii_letters, digits
import os
import sys 
import time
from psychopy.hardware import keyboard
from questplus import QuestPlus
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

####### EXPERIMENT PARAMETERS ####################################################################################################################################################################################################

EYETRACKER_OFF = False # Set this variable to True to run the script without eyetracking

TRIAL_REPETITIONS = 16 # How many times to repeat each of the 12 unique trial_types (total # trials = TRIAL_REPEITIONS * length(trial_types))
PRACTRIALS_REPETITIONS = 1 # Same as above, but for practice trials
MAX_REPEATS = 3 # Maximum number of consecutive trials of the same cue condition (valid, invalid, neutral)
CUE_SIZE = [.5, .5] # deg
TARGET_SIZE = [1.5, 1.5] # deg
FIXATION_SIZE = [.5, .5] # deg
POSITION = np.array([5.0, 0.0]) # 5DVA eccentricity for target and cues
GAZE_THRESHOLD = 2.5 # deg; if gaze shifts more than this from center, trial is aborted 
SPATIAL_FREQUENCY = 5
PRACT_CONTRASTS = [0.1, 0.5, 1.0] * 4  # Hardcoded contrast values for practice trials

# Timing (s)
frameTolerance = 0.001  # How close to onset before 'same' frame
ITI = 2.5 # Duration of fixation point between trials (s)
CUE_DURATION = 0.05 # (s)
ISI = 0.1 # Duration of fixation point between cue and target (s)
TARGET_DURATION = 0.5 # (s)
RESPONSE_DURATION = 1.0 # (s) Total response window is TARGET_DURATION + RESPONSE_DURATION 
TOTAL_TRIAL_DURATION = ITI + CUE_DURATION + ISI + TARGET_DURATION + RESPONSE_DURATION
MAX_PRESENTATIONS = 3 # Maximum number of times each trial can be presented
FEEDBACK_DURATION = 1.0 # (s)
GAZE_CHECK = [ITI, ITI + CUE_DURATION+ISI+TARGET_DURATION] # From cue onset to target offset

# Create a dictionary of 12 unique trial types used to create practice and experiment trial lists
trial_types = data.createFactorialTrialList({
            'orientation': [0, 90], # 0 - vertical; 90 - horizontal
            'gabor_position': [-1, 1], # -1 = Left, 1 = Right
            'cue_condition': ['Neutral', 'Invalid','Valid']  
            }) 

####### WINDOW, DATA FILE, & EYETRACKER SETUP ####################################################################################################################################################################################################

# Eyetracker setup is based off of sample scripts provided by SR Research for psychopy coder

# Collect participant ID, session number, number of blocks
exp_name = 'EnhancementTask'
exp_info = {
    'Participant ID': '',
    'Session': '',
    'Blocks':'Ex. 4,6,8'}

while True:
    dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name)
    if dlg.OK == False:
        core.quit()
        sys.exit()

    blocks = int(exp_info['Blocks'])
    # Clean participant ID
    participant_id = exp_info['Participant ID'].rstrip().split(".")[0]
    edf_filename = f"{participant_id}_ET"

    # check if the filename and number of blocks are valid
    allowed_char = ascii_letters + digits + '_'
    if not all([c in allowed_char for c in edf_filename]):
        raise ValueError('ERROR: Invalid EDF filename. Enter only letters, digits, or underscores.')
    elif len(edf_filename) > 8:
        raise ValueError("ERROR: Invalid EDF filename: must be ≤5 characters.")
    elif (TRIAL_REPETITIONS*len(trial_types))%blocks != 0:
        raise ValueError("ERROR: Invalid number of blocks. Must be a factor of 192.")
    else:
        break

BREAK_INTERVAL = (TRIAL_REPETITIONS*len(trial_types))/blocks # Number of trials before each break
print("Break interval:", BREAK_INTERVAL)

# Establish data output directory
time_str = time.strftime("_%m_%d_%Y", time.localtime())
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"{participant_id}_{exp_name}_Session{exp_info['Session']}{time_str}")
os.makedirs(output_folder, exist_ok=True)
edf_path = os.path.join(output_folder, edf_filename + '.EDF')

# Set file name for psychopy task data (.csv, .psydat)
filename = os.path.join(output_folder, f"{participant_id}_{exp_name}_Session{exp_info['Session']}") 

# Create a file to log messages and warnings
logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # Warnings, errors, and critical messages will be displayed in output console

# Connect to the EyeLink Host PC
if EYETRACKER_OFF:
    el_tracker = pylink.EyeLink(None) # Need this line so code will run when not using eyetracking
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print("Error #1") # troubleshoot
        print('ERROR:', error)
        core.quit()
        sys.exit()
        
# Open the EDF data file on the Host PC
edf_file = edf_filename + ".EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print("Error #2") #troubleshoot
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

# Window setup for EIZO monitor
Eizo = monitors.Monitor('Eizo', width = 51.84, distance = 60)
Eizo.setSizePix([1920, 1200])
win = visual.Window(fullscr=True, color=[0,0,0],
            size=Eizo.getSizePix(), screen=1,
            winType='pyglet', allowStencil=False,
            monitor=Eizo, colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=False,
            units='deg', 
            checkTiming=False)

# Get the screen resolution used by PsychoPy
scn_width, scn_height = win.size

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
genv.setCalibrationSounds('', '', '')

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)
logging.info(f"Graphics environment set up: {genv}")

# Store frame rate
exp_info['frameRate'] = win.getActualFrameRate()
if exp_info['frameRate'] is not None:
    frameDur = 1.0 / round(exp_info['frameRate'])
else:
    frameDur = 1.0 / 60.0 # couldn't get a reliable measure so guess
    logging.warning('Frame rate is unknown. Using frame duration of 1/60s.')
exp_info['frameDur'] = frameDur   

# Create an experiment handler
thisExp = data.ExperimentHandler(
    name=exp_name, version='',
    extraInfo=exp_info, runtimeInfo=None,
    originPath=os.path.abspath(__file__),
    savePickle=True, saveWideText=True,
    dataFileName=filename, 
    sortColumns='time') 

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

####### INITIALIZE VISUAL COMPONENTS #################################################################################################################################################################################################### 

kb = keyboard.Keyboard()

# --- Initialize components for Welcome Screen ---
welcome_text = visual.TextStim(win=win, name='welcome_text',
    text='''Welcome to the Line Grate Game!''',
    font='Arial', units='deg', 
    pos=(0, 0), draggable=False, height=1.5, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Instructions ---
instruct_text = visual.TextStim(win=win, name='instruct_text',
    text='''In this game, you will see line grates like these:\n\n\n\n
    Your job is to determine which way the lines are pointing!\n\nPress the left button if the lines are pointing up and down, 
    and the right button if the lines are pointing side to side.''',
    font='Arial', units='height', 
    pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
gabor_inst1 = visual.GratingStim(
    win=win, name='gabor_inst1',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=90, pos=[5,2], size=[4,4], sf=[SPATIAL_FREQUENCY], phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
gabor_inst2 = visual.GratingStim(
    win=win, name='gabor_inst2',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=0.0, pos=[-5,2], size=[4,4], sf=[SPATIAL_FREQUENCY], phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)

# --- Initialize components for Trials ---
fix_cross = visual.ShapeStim(
    win=win, name='fix_cross', vertices='cross',units='deg', 
    size=(FIXATION_SIZE[0], FIXATION_SIZE[1]),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=0.0, interpolate=True)
left_cue = visual.ShapeStim(
    win=win, name='left_cue',units='deg', 
    size=[CUE_SIZE[0], CUE_SIZE[1]], vertices='circle',
    ori=0.0, pos=[-POSITION[0], POSITION[1]], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=-1.0, interpolate=True)
right_cue = visual.ShapeStim(
    win=win, name='right_cue',units='deg', 
    size=[CUE_SIZE[0], CUE_SIZE[1]], vertices='circle',
    ori=0.0, pos=[POSITION[0], POSITION[1]], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=1.0, depth=-1.0, interpolate=True)
gabor = visual.GratingStim(
    win=win, name='gabor',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=0.0, pos=[POSITION[0],POSITION[1]], size=[TARGET_SIZE[0], TARGET_SIZE[1]], sf=[SPATIAL_FREQUENCY], phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True)
feedback_text = visual.TextStim(win=win, name='feedback_text',
    text="", font='Arial',
    units='deg', pos=(0, -2.5), draggable=False, height=1.2, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',depth=0.0);
feedback_image = visual.ImageStim(win=win,
    name='feedback_image', units='deg', 
    image='sin', mask=None,
    ori=0, pos=(0, 2.5), size=(5, 5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
gaze_feedback = visual.TextStim(win=win, name='gaze_feedback',
    text="", font='Arial',
    units='deg', pos=(0, -2.5), draggable=False, height=1.2, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',depth=0.0);
prac_outcome_text = visual.TextStim(win=win, name='prac_outcome_text',
    text="", font ='Arial', color= 'black',
    units='deg', pos=(0, 0), draggable=False, height=1.2, wrapWidth=1700, ori=0)
break_text = visual.TextStim(win=win, name='break_text',
    text="Great job!\nLet's take a quick break!", font ='Arial', color= 'black',
    units='deg', pos=(0, 0), draggable=False, height=1.2, wrapWidth=1700, ori=0)

# --- Initialize components for the End ---
end_text = visual.TextStim(win=win, name='end_text',
    text="You finished the game!",
    font='Arial', units = 'deg',
    pos=(0, 0), draggable=False, height=1.5, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

####### FUNCTIONS #################################################################################################################################################################################################### 

# ------------- Functions for Eyetracker from sample scripts (with some adjustments)
def abort_trial():
    # Returns None and clears the eyetracker
    el_tracker = pylink.getEYELINK()
    
    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # Clear the psychopy window
    if win is not None:
        win.clearAutoDraw()
        win.flip()
    
    # Send a message to clear the Data Viewer screen
    el_tracker.sendMessage('!V CLEAR 116 116 116')

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
    
    if thisExp.status != FINISHED:
        print('Trial aborted.')
        trial['presented'] += 1
        thisExp.addData('Presentations', trial['presented'])
        thisExp.addData('trial.aborted', 'aborted')
        thisExp.nextEntry()

    return None
    
def terminate_task():
    """ Disconnects from the eyetracker and saves all data files """
    
    # Save task data
    thisExp.saveAsWideText(thisExp.dataFileName + '.csv', delim='auto')
    thisExp.saveAsPickle(thisExp.dataFileName)
    logging.flush()
    
    # Clear the psychopy window
    if win is not None:
        win.clearAutoDraw()
        win.flip()
        
    # Mark experiment as finished
    thisExp.status = FINISHED
    thisExp.abort()

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
        print('Task terminated. EDF data is transferring from EyeLink Host PC...')

        # Download the EDF data file from the Host PC to a local data folder
        try:
            el_tracker.receiveDataFile(edf_filename, edf_path)
            print(f"EDF file saved to: {edf_path}")
        except RuntimeError as error:
            print('ERROR downloading EDF file:', error)

        # Close the link to the tracker
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()

# ------------- Functions to run the task

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
    if not isinstance(comp, keyboard.Keyboard):  # kb is handled differently
        thisExp.timestampOnFlip(win, f'{comp.name}.started')
        comp.setAutoDraw(True)
    else:
        thisExp.timestampOnFlip(win, 'kb.started')

def erase_comp(comp, t, tThisFlipGlobal, frameN):
    comp.tStop = t
    comp.tStopRefresh = tThisFlipGlobal
    comp.frameNStop = frameN
    comp.status = FINISHED
    if not isinstance(comp, keyboard.Keyboard):
        thisExp.timestampOnFlip(win, f'{comp.name}.stopped')
        comp.setAutoDraw(False)
    else:
        thisExp.timestampOnFlip(win, 'kb.stopped')

def get_eye_used(el_tracker):
    if EYETRACKER_OFF:
        return 0 # For when running without eyetracking

    # Returns 0 for left, 1 for right, None if eye data cannot be collected. 
    if el_tracker is not None and el_tracker.isConnected():
        eye = el_tracker.eyeAvailable()
        if eye in [0, 1]:
            el_tracker.sendMessage(f"EYE_USED {eye} {'RIGHT' if eye == 1 else 'LEFT'}")
            return eye
        elif eye == 2: # bonicular vision defaults to left eye
            el_tracker.sendMessage("EYE_USED 0 LEFT")
            return 0
            
    print("ERROR: EyeLink not connected or invalid eye")
    return None

def is_gaze_within_bounds(el_tracker, eye_used, loss_clock, bounds_deg=GAZE_THRESHOLD, loss_threshold=0.1):
    while True:
        sample = el_tracker.getNewestSample()

        if eye_used == 0 and sample.isLeftSample():
            eye_data = sample.getLeftEye()
        elif eye_used == 1 and sample.isRightSample():
            eye_data = sample.getRightEye()
        else:
            if loss_clock.getTime() > loss_threshold:
                print("Loss of sample for longer than 100ms.")
                thisExp.addData('Loss clock', loss_clock)
                return False  # No valid eye data in this sample
            continue

        gaze = eye_data.getGaze()
        pupil = eye_data.getPupilSize()

        # Blink check: pupil size is 0 or negative
        if pupil <= 0:
            print("Blink detected.")
            return True
        
        loss_clock.reset()

        # Convert to degrees
        dx = gaze[0] - scn_width / 2
        dy = scn_height / 2 - gaze[1]
        pixels = np.array([dx, dy])
        dx_deg, dy_deg = pix2deg(pixels, monitor=Eizo)
    #    print(f"Gaze in degrees: ({dx_deg},{dy_deg})")
    
        return abs(dx_deg) <= bounds_deg and abs(dy_deg) <= bounds_deg

def run_trial(trial, trial_index, practice = False):
    # Returns response: 1- Correct; 0 - Incorrent; None - No response or trial aborted
    continueRoutine = True
    kb.keys = []
    kb.rt = []
    kb_allKeys = []
    eye_used = None

    # Initialize the clock that will keep track of signal loss duration
    loss_clock = core.Clock()

    components = [fix_cross, left_cue, right_cue, gabor, kb]
    
    # Esure tracker is ready to receive commands
    el_tracker = pylink.getEYELINK()
    el_tracker.setOfflineMode()
    
    # When using eyetracking, abort trial if no eye information can be collected
    eye_used = get_eye_used(el_tracker)
    if eye_used is None and not EYETRACKER_OFF:
        return abort_trial()
    
    # send a "TRIAL" message to mark the start of a trial and send status message to host pc
    el_tracker.sendMessage('TRIALID %d' % trial_index)
    status_msg = 'TRIAL number %d' % trial_index
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)

    # put tracker in idle/offline mode before recording
    el_tracker.setOfflineMode()
    
    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link, event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print("Error #4") # troubleshoot
        print("ERROR:", error)
        return abort_trial()
    
    pylink.pumpDelay(100) # add 100 msec to catch final events before starting the trial

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

    gabor.pos = np.array([POSITION[0] * trial['gabor_position'], POSITION[1]])
    gabor.ori = trial['orientation']
    left_cue.opacity, right_cue.opacity = get_cue_opacity(trial['cue_condition'], trial['gabor_position'])
    if trial['gabor_position'] == -1:
        gabor_pos_readable = 'L'
    else:
        gabor_pos_readable = 'R'

    if practice: # Set Gabor contrast and orientation for practice trials
        intensity = pract_contrasts.pop(0)
        gabor.contrast = intensity
    
    if not practice: # Questplus algorithm will determine gabor contrast in exp trials
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
        
        # Update gabor contrast
        gabor.contrast = intensity

    while continueRoutine: # Update components on each frame
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  
    
        if fix_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            fix_cross.color = 'white'
            draw_comp(fix_cross, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('fixation_started')
        
        if fix_cross.status == STARTED:
            # Make fixation cross green for 500ms in the middle of ITI
            if tThisFlip >= 1.0-frameTolerance and tThisFlip < ITI-frameTolerance:
                fix_cross.color = 'green'
            if tThisFlip >= 1.5-frameTolerance and tThisFlip < ITI-frameTolerance:
                fix_cross.color = 'white'
            if tThisFlipGlobal > fix_cross.tStartRefresh + TOTAL_TRIAL_DURATION-frameTolerance:
                erase_comp(fix_cross, t, tThisFlipGlobal, frameN)
        
        if left_cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
            fix_cross.color = 'white'
            draw_comp(left_cue, t, tThisFlipGlobal, frameN)
            draw_comp(right_cue, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('cue_started')
        
        if left_cue.status == STARTED and tThisFlipGlobal > left_cue.tStartRefresh + CUE_DURATION-frameTolerance:
            erase_comp(left_cue, t, tThisFlipGlobal, frameN)
            erase_comp(right_cue, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('cue_stopped')
            
        # Gaze contingency 
        if not EYETRACKER_OFF and GAZE_CHECK[0]-frameTolerance <= tThisFlip <= GAZE_CHECK[1]-frameTolerance:
            if not is_gaze_within_bounds(el_tracker, eye_used, loss_clock = loss_clock):
                return abort_trial()
        # -------------------------------------------------------------------------------------------------

        if gabor.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
            draw_comp(gabor, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('target_started')

        if gabor.status == STARTED and tThisFlipGlobal > gabor.tStartRefresh + TARGET_DURATION-frameTolerance:
            erase_comp(gabor, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('target_stopped')
        
        waitOnFlip = False

        if kb.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
            draw_comp(kb, t, tThisFlipGlobal, frameN)
            waitOnFlip = True 
            win.callOnFlip(kb.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(kb.clearEvents, eventType='keyboard')
        
        if kb.status == STARTED and tThisFlipGlobal > kb.tStartRefresh + (TARGET_DURATION+RESPONSE_DURATION)-frameTolerance:
            erase_comp(kb, t, tThisFlipGlobal, frameN)

        if kb.status == STARTED and not waitOnFlip: 
            theseKeys = kb.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
            kb_allKeys.extend(theseKeys)
            if len(kb_allKeys):
                thisExp.timestampOnFlip(win, 'kb.stopped') # record time of key press
                kb.keys = kb_allKeys[-1].name  # just the last key pressed
                kb.rt = kb_allKeys[-1].rt
                kb.duration = kb_allKeys[-1].duration
                continueRoutine = False  # end the routine on key press
    
        if kb.getKeys(keyList=["escape"]):
            terminate_task()

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
    thisExp.addData('gabor.intensity', intensity)
    thisExp.addData('gabor.pos', gabor.pos)
    thisExp.addData('gabor.pos.readable', gabor_pos_readable)
    thisExp.addData('gabor.ori', gabor.ori)
    thisExp.addData('left_cue.opacity', left_cue.opacity)
    thisExp.addData('right_cue.opacity', right_cue.opacity)
    thisExp.addData('Condition', trial['cue_condition'])
    
    # Send trial data to EDF file
    el_tracker.sendMessage('!V TRIAL_VAR condition %s' % trial['cue_condition'])
    try:
        el_tracker.sendMessage('!V TRIAL_VAR RT %d' % int(kb.rt))
    except (TypeError, ValueError):
        el_tracker.sendMessage('!V TRIAL_VAR RT -1') # If no response, RT is set to -1 in eyetracker data

    if not practice:
        thisExp.addData('QP_Threshold', threshold)
        thisExp.addData('QP_Slope', slope)
        thisExp.addData('QP_Lapse', lapse_rate)

    # Response Check
    if kb.keys in ['', [], None]:  # No response was made
        kb.keys = None
        response = None
        if practice:
            feedback_text.text = "Remember to press a button!"
            feedback_image.setImage("images/x_mark.png")
            feedback_image.draw()
            feedback_text.draw()
            win.flip()
            core.wait(FEEDBACK_DURATION)
    else:
        response = 1 if (
            (kb.keys == '1' and trial['orientation'] == 0) or 
            (kb.keys == '2' and trial['orientation'] == 90)
            ) else 0
        if not practice:
            current_qp.update(stim={'intensity': intensity}, outcome={'response': response})
        if response == 1:
            if practice:
                feedback_text.text = "Correct!"
                feedback_image.setImage("images/check_mark.png")
                feedback_image.draw()
                feedback_text.draw()
                win.flip()
                core.wait(FEEDBACK_DURATION)
        else:
            if practice:
                feedback_text.text = "Try Again!"
                feedback_image.setImage("images/x_mark.png")
                feedback_image.draw()    
                feedback_text.draw()
                win.flip()
                core.wait(FEEDBACK_DURATION)         

    thisExp.addData('Keypress', kb.keys)
    thisExp.addData('Accuracy', response)
    # Save response, accuracy, RT, and duration to data file and print response
    if kb.keys != None: 
        thisExp.addData('RT', kb.rt)
    
    print("Response:", response)
    print(f"Next Intensity: {intensity}")
    
    el_tracker.sendMessage('!V CLEAR 128 128 128')
    
    # Stop recording between trials to decrease size of output file
    pylink.pumpDelay(100) # add 100 msec to catch final events before stopping
    el_tracker.stopRecording()
    
    # Send trial result message to mark the end of the trial
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    
    trial['presented'] += 1
    thisExp.addData('Presentations', trial['presented'])
    
    thisExp.nextEntry() # Advance to the next row in the data file
        
    return response
    
####### WELCOME SCREEN AND CALIBRATION  #################################################################################################################################################################################################### 

# Set clocks and start experiment
globalClock = core.Clock()
routineTimer = core.Clock()
logging.setDefaultClock(globalClock)
win.flip()

exp_info['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
thisExp.status = STARTED

# Window needs to be flipped before trying to run calibration or else code won't run
welcome_text.draw()
win.flip()

# Calibrate eyetracker
if not EYETRACKER_OFF:
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print("ERROR #5") #troubleshoot
        print('ERROR:', err)
        el_tracker.exitCalibration()

####### INSTRUCTIONS  #################################################################################################################################################################################################### 

instruct_text.draw()
gabor_inst1.draw()
gabor_inst2.draw()
win.flip()
thisExp.addData('Instructions.started', globalClock.getTime(format='float'))

if kb.getKeys(keyList=["escape"]):
    terminate_task()

event.waitKeys(keyList=['space']) #Press space to continue to practice trials
thisExp.addData('Instructions.stopped', globalClock.getTime(format='float'))
thisExp.nextEntry()

####### PRACTICE BLOCK #################################################################################################################################################################################################### 

repeat_count = 0
repeat_practice = True
while repeat_practice:
    repeat_count += 1
    total_correct = 0

    trial_list = get_trials(PRACTRIALS_REPETITIONS)
    
    for trial in trial_list:
        trial['presented'] = 0

    pract_contrasts = PRACT_CONTRASTS.copy() 
    shuffle(pract_contrasts)

    for trial in trial_list:
        routineTimer.reset()
        thisExp.addData('Trial', trial['Index'])
        print(f"Practice Trial {trial['Index']}:", trial)
        answer = run_trial(trial, trial['Index'], practice=True)

        if answer != None:
            total_correct += answer

    thisExp.addData('Practice.stopped', globalClock.getTime(format='float'))

    # Calculate accuracy on practice trials, determine whether to continue to experiment block, repeat practice, or end the experiment
    percent_correct = (total_correct / (len(trial_list))) * 100
    print(f"Percent correct: {percent_correct}")

    if percent_correct >= 75: # Continue to experiment block
        repeat_practice = False
        prac_outcome_text.text = "Great Job!\n\nAre you ready to play the real game?"
        print("Practice block passed! Starting experiment trials.")
        prac_outcome_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
    else:
        if repeat_count < 2: # Repeat practice block if less than 75% accuracy on first try
            prac_outcome_text.text = "Let's try the practice again!"
            prac_outcome_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
#            routineTimer.reset()
        else: # End experiment if accuracy below 75% after two attempts
            print("Participant did not pass practice trials. Ending experiment.")
            end_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            terminate_task()

####### EXPERIMENT BLOCK #################################################################################################################################################################################################### 

thisExp.addData('Experiment.started', globalClock.getTime(format='float'))

no_resp_trials = []
total_trials = 0

trial_list = get_trials(TRIAL_REPETITIONS)
for trial in trial_list:
    trial['presented'] = 0 # start keeping track of how many times each trial has been presented 

for trial in trial_list:
    routineTimer.reset()
    total_trials += 1
    thisExp.addData('Trial', trial['Index'])

    answer = run_trial(trial, trial['Index'])
    print(f"Experiment Trial {trial['Index']}:", trial)

    if answer is None:
        no_resp_trials.append(trial)  

    # Give break every interval and do a drift check to recalibrate if necessary
    if total_trials % BREAK_INTERVAL == 0: 
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
                
#        routineTimer.reset()

# Repeat trials with no response
while len(no_resp_trials) > 0:
    print(f"Re-running {len(no_resp_trials)} trials with no response...")

    remaining_trials = []

    for trial in no_resp_trials:
        routineTimer.reset()
        trial_num = trial_list.index(trial) + 1
        thisExp.addData('Trial', trial_num)

        answer = run_trial(trial, trial['Index'])
        print(f"Experiment Trial {trial['Index']}:", trial)

        if answer is None and trial['presented'] < MAX_PRESENTATIONS:
            remaining_trials.append(trial)

    no_resp_trials = remaining_trials

####### END EXPERIMENT #################################################################################################################################################################################################### 

end_text.draw()
win.flip()
event.waitKeys(keyList=['space'])

terminate_task() # This function handles data saving