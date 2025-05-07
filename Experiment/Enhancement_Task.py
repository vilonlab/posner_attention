# --- Import packages ---
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)
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

####### EYETRACKER SETUP ####################################################################################################################################################################################################
EYETRACKER_OFF = False #Set this variable to True to run the script without eyetracking

# Collect participant ID and session number
exp_name = 'Enhancement_Task'
exp_info = {
    'Participant ID': '',
    'Session': ''}

dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name)
if dlg.OK == False:
    core.quit()

# Clean participant ID
participant_id = exp_info['Participant ID'].rstrip().split(".")[0]

# Make a subfolder within the data folder
_thisDir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_thisDir, 'data')
participant_folder = f"{participant_id}" + '_ET' # name of subfolder within data folder
output_folder = os.path.join(data_dir, participant_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# Name, sanitize, and validate EDF file (length <= 8 & no special characters)
edf_filename = participant_id + '_ET'
edf_path = os.path.join(output_folder, edf_filename + '.EDF')
allowed_char = ascii_letters + digits + '_'
if not all([c in allowed_char for c in edf_filename]):
    raise ValueError("Invalid EDF filename: Only letters, digits, and underscores are allowed.")
elif len(edf_filename) > 8:
    raise ValueError("Invalid EDF filename: Must be 8 characters or fewer.")

# Get timestamp
time_str = time.strftime("_%m_%d_%Y", time.localtime())
session_identifier = f"{edf_filename}_Session{exp_info['Session']}{time_str}"

# Set file name for task data
filename = os.path.join(output_folder, f"{participant_id}_{exp_name}")  # for .csv, .log, .psydat
logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

# Connect to the EyeLink Host PC
if EYETRACKER_OFF:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()
        
# Open an EDF data file on the Host PC
edf_file = edf_filename + ".EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add text for data viewing
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

eyelink_ver = 0  # set version to 0, in case running without eyetracking
if not EYETRACKER_OFF:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])
    # print out some version info in the shell
    print('Running experiment on %s, version %d' % (vstr, eyelink_ver))
    
# what eye events to save in the EDF file, include everything by default
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
# what eye events to make available over the link, include everything by default
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
# what sample data to save in the EDF data file and to make available over the link
if eyelink_ver > 3:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
else:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Choose a calibration type (HV = horizontal/vertical), choosing 9-point calibration
el_tracker.sendCommand("calibration_type = HV9")

# Window setup for EIZO monitor
win = visual.Window(fullscr=True, color=[0,0,0],
            size=[1920, 1200], screen=1,
            winType='pyglet', allowStencil=False,
            monitor='Eizo', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False)

# get the native screen resolution used by PsychoPy
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

# Set background and foreground colors for the calibration target
foreground_color = (-1, -1, -1)
background_color = win.color
genv.setCalibrationColors(foreground_color, background_color)
genv.setTargetType('picture')
genv.setPictureTarget(os.path.join(_thisDir, 'Images', 'andy_fixation.png')) # need to check the size of Andy

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('', '', '')

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)

# Calibrate eyetracker
if not EYETRACKER_OFF:
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()
    Should_recal = 'no'

####### EXPERIMENT SETUP ####################################################################################################################################################################################################

# Create an experiment handler to help with data saving
thisExp = data.ExperimentHandler(
    name=exp_name, version='',
    extraInfo=exp_info, runtimeInfo=None,
    originPath=os.path.abspath(__file__),
    savePickle=True, saveWideText=True,
    dataFileName=filename, 
    sortColumns='time')

# Store frame rate
exp_info['frameRate'] = win.getActualFrameRate()
if exp_info['frameRate'] is not None:
    frameDur = 1.0 / round(exp_info['frameRate'])
else:
    frameDur = 1.0 / 60.0 # couldn't get a reliable measure so guess
    logging.warning('Frame rate is unknown. Using frame duration of 1/60s.')
exp_info['frameDur'] = frameDur    

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
frameTolerance = 0.001  # how close to onset before 'same' frame
ITI = 1.0 # fixation point between trials
CUE_DURATION = 0.05 
ISI = 0.1 # fixation point between cue and target
TARGET_DURATION = 1.0 
RESPONSE_DURATION = 1.0 # total response window is TARGET_DURATION + RESPONSE_DURATION 
TOTAL_TRIAL_DURATION = ITI + CUE_DURATION + ISI + TARGET_DURATION + RESPONSE_DURATION
BREAK_INTERVAL = 32 # Number of trials before each break
MAX_PRESENTATIONS = 3 # Maximum number of times each trial can be presented 

####### INITIALIZE VISUALS #################################################################################################################################################################################################### 
kb = keyboard.Keyboard()

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
    units='height', pos=(0, 0.15), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
feedbackImage = visual.ImageStim(win=win,
    name='feedbackImage', units='height', 
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(.2, .2),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
break_text = visual.TextStim(win=win, name='break_text',
    text="Great job!\nLet's take a quick break!", font ='Arial', color= 'black',
    units='height', pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0)

# --- Initialize components for the End ---
end_text = visual.TextStim(win=win, name='end_text',
    text="You finished the game!",
    font='Arial',
    pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

####### FUNCTIONS #################################################################################################################################################################################################### 

# ------------- Functions for Eyetracker from sample scripts
def clear_screen(win):
    """ clear up the PsychoPy window"""
    win.fillColor = genv.getBackgroundColor()
    win.flip()
    
def abort_trial():
    """Ends recording """
    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()
        
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR
    
def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
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
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        try:
            el_tracker.receiveDataFile(edf_filename, edf_path)
            print(f"EDF file saved to: {edf_path}")
        except RuntimeError as error:
            print('ERROR downloading EDF file:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()

# ------------- Functions to run the task by VBG

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

def run_trial(trial, trial_index, practice = False):
    continueRoutine = True
    kb.keys = []
    kb.rt = []
    kb_allKeys = []

    components = [Fixation_Point, Left_Cue, Right_Cue, Gabor, kb]
    
    # Esure tracker is ready to receive commands
    el_tracker = pylink.getEYELINK()
    el_tracker.setOfflineMode()
    
    # Draw fixation cross on host pc to ensure gaze location
    cross_coords = (int(scn_width/2.0), int(scn_height/2.0))
#    el_tracker.sendCommand('clear_screen 0')
    el_tracker.sendCommand('draw_cross %d %d 10' % cross_coords)  # draw cross
    
    # send a "TRIAL" message to mark the start of a trial and send status message to host pc
    el_tracker.sendMessage('TRIALID %d' % trial_index)
    status_msg = 'TRIAL number %d' % trial_index
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)
    
    # drift check
    # we recommend drift-check at the beginning of each trial
    # the doDriftCorrect() function requires target position in integers
    # the last two arguments:
    # draw_target (1-default, 0-draw the target then call doDriftCorrect)
    # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
    while not EYETRACKER_OFF:
        # terminate the task if no longer connected to the tracker
        if (not el_tracker.isConnected()) or el_tracker.breakPressed():
            terminate_task()
            return pylink.ABORT_EXPT

        # drift-check and re-do camera setup if ESCAPE is pressed
        try:
            error = el_tracker.doDriftCorrect(int(scn_width/2.0),int(scn_height/2.0), 1, 1)
            # break following a success drift-check
            if error is not pylink.ESC_KEY:
                break
        except:
            pass
            
    # put tracker in idle/offline mode before recording
    el_tracker.setOfflineMode()
    
    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link, event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial()
        return pylink.TRIAL_ERROR

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
            el_tracker.sendMessage('fixation_started')
        
        if Fixation_Point.status == STARTED:
            if tThisFlip >= 0.9-frameTolerance and tThisFlip < ITI-frameTolerance:
                Fixation_Point.color = 'green'
            if tThisFlipGlobal > Fixation_Point.tStartRefresh + TOTAL_TRIAL_DURATION-frameTolerance:
                erase_comp(Fixation_Point, t, tThisFlipGlobal, frameN)
        
        if Left_Cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
            Fixation_Point.color = 'white'
            draw_comp(Left_Cue, t, tThisFlipGlobal, frameN)
            draw_comp(Right_Cue, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('cue_started')
        
        if Left_Cue.status == STARTED and tThisFlipGlobal > Left_Cue.tStartRefresh + CUE_DURATION-frameTolerance:
            erase_comp(Left_Cue, t, tThisFlipGlobal, frameN)
            erase_comp(Right_Cue, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('cue_stopped')

        if Gabor.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
            draw_comp(Gabor, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('target_started')

        if Gabor.status == STARTED and tThisFlipGlobal > Gabor.tStartRefresh + TARGET_DURATION-frameTolerance:
            erase_comp(Gabor, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('target_stopped')
            Fixation_Point.color = 'blue'
        
        waitOnFlip = False  # Wait for key response

        if kb.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
            draw_comp(kb, t, tThisFlipGlobal, frameN)
            waitOnFlip = True  # Wait for key response
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
            thisExp.status = FINISHED
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
    thisExp.addData('Gabor.intensity', intensity)
    thisExp.addData('Gabor.pos', Gabor.pos)
    thisExp.addData('Gabor.ori', Gabor.ori)
    thisExp.addData('Left_Cue.opacity', Left_Cue.opacity)
    thisExp.addData('Right_Cue.opacity', Right_Cue.opacity)
    thisExp.addData('Condition', trial['cue_condition'])
    
    # Send trial data to EDF file
    el_tracker.sendMessage('!V TRIAL_VAR condition %s' % trial['cue_condition'])
    el_tracker.sendMessage('!V TRIAL_VAR RT %d' % kb.rt)

    if not practice:
        thisExp.addData('QP_Threshold', threshold)
        thisExp.addData('QP_Slope', slope)
        thisExp.addData('QP_Lapse', lapse_rate)

    # Response Check
    if kb.keys in ['', [], None]:  # No response was made
        kb.keys = None
        response = None
        if practice:
            feedback.text = "Remember to press a button!"
            feedbackImage.setImage("Images/x_mark.png")
            feedbackImage.draw()
            feedback.draw()
            win.flip()
            core.wait(1)
    else:
        response = 1 if (
            (kb.keys == '1' and trial['orientation'] == 0) or 
            (kb.keys == '2' and trial['orientation'] == 90)
            ) else 0
        if not practice:
            current_qp.update(stim={'intensity': intensity}, outcome={'response': response})
        if response == 1:
            if practice:
                feedback.text = "Correct!"
                feedbackImage.setImage("Images/check_mark.png")
                feedbackImage.draw()
                feedback.draw()
                win.flip()
                core.wait(1)
        else:
            if practice:
                feedback.text = "Try Again!"
                feedbackImage.setImage("Images/x_mark.png")
                feedbackImage.draw()    
                feedback.draw()
                win.flip()
                core.wait(1)         

    thisExp.addData('Keypress',kb.keys)
    thisExp.addData('Accuracy', response)
    # Save response, accuracy, RT, and duration to data file and print response
    if kb.keys != None: 
        thisExp.addData('RT', kb.rt)
        thisExp.addData('Keypress Duration', kb.duration)
    
    print("Accuracy:", response)

    routineTimer.reset()  # Reset the routine timer for the next trial
    thisExp.nextEntry() # Advance to the next row in the data file
    
    el_tracker.sendMessage('!V CLEAR 128 128 128')
    
    # Stop recording between trials to decrease size of output file
    pylink.pumpDelay(100) # add 100 msec to catch final events before stopping
    el_tracker.stopRecording()
    
    # Send trial result message to mark the end of the trial
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
        
    return response

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

if kb.getKeys(keyList=["escape"]):
    thisExp.status = FINISHED
    terminate_task()

event.waitKeys(keyList=['space']) #Press space to continue to practice trials
thisExp.addData('Instructions.stopped', globalClock.getTime(format='float'))
thisExp.nextEntry()

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
        answer = run_trial(trial, trial['Index'], practice=True)

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
            terminate_task()

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

        if answer is None and trial['presented'] <= MAX_PRESENTATIONS:
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

terminate_task()