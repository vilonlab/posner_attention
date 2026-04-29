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
routineTimer = core.Clock() # initialize trial-level clock (will get reset at the start of each trial)

EYETRACKER_OFF = True # Set to True to run the script without eyetracking
RESPONSE_KEYS = ['1', '2'] # 1 for left, 2 for right

# get all 8 unique trial types by combining the trial variables (e.g., one trial type is: {'orientation': 0,  'gabor_position': -1, 'freq_condition': 'High'})
TRIAL_TYPES = data.createFactorialTrialList({
            'orientation': [0, 90], # 0 - vertical; 90 - horizontal
            'gabor_position': [-1, 1], # -1 = Left, 1 = Right
            'freq_condition': ['High', 'Low']  
            })

# practice blocks
PTRIAL_PRESENTATIONS = 2 # how many times to present each of the 8 unique TRIAL_TYPES in practice blocks 2 and 3
PTOTAL_TRIALS = PTRIAL_PRESENTATIONS * len(TRIAL_TYPES) # total trials in practice blocks 2 and 3
PRACT1_CONTRASTS = 1.0 # 100% contrast for all trials in practice block 1
PRACT_CONTRASTS = [0.1, 0.4, 0.7, 1.0] # hardcoded possible gabor contrast values for practice trials; keep length to a factor of 8
EXTENDED_TARGET_DUR = 0.5 # target duration for practice block 2
ACCURACY_THRESHOLD = 75 # accuracy needed to pass the practice blocks
MAX_PRACTICE_REPEATS = 2 # maximum number of times each practice block can be repeated before experiment ends

# experiment blocks
TRIAL_PRESENTATIONS = 16 # how many times to present each of the 8 unique TRIAL_TYPES throughout all experiment blocks (64 trials for each frequency condition)
TOTAL_TRIALS = TRIAL_PRESENTATIONS * len(TRIAL_TYPES) # total number of experiment trials
MAX_CONSECUTIVE_TRIALS = 3 # maximum number of consecutive trials of the same sf condition (high, low)
MAX_TRIAL_REPEATS = 3 # maximum number of times each trial can be presented after no response (includes initial presentation)

# stims (deg)
TARGET_SIZE = 2
FIXCROSS_SIZE = 0.75
ANDYFIX_SIZE = 1.5
POSITION = np.array([6.0, 0.0]) # DVA eccentricity for target
HIGH_SPATIAL_FREQ = 8
LOW_SPATIAL_FREQ = 2

# timing (s)
frameTolerance = 0.001  # How close to onset before 'same' frame
FIX_CROSS_DUR = 1.0 # s; duration of fixation cross at start of trial
ANDY_FIX_DUR = 0.5 # s; Andy to signal start of trial
EXP_TARGET_DUR = 0.1 # s; target gabor duration for experiment trials
RESPONSE_WINDOW = 2.0 # s; duration of andy fixation after target offset; total response window is TARGET_DUR + RESPONSE_WINDOW
FEEDBACK_DUR = 1 # s; duration of feedback presentation for practice blocks

####### WINDOW, DATA FILE, & EYETRACKER SETUP ####################################################################################################################################################################################################

# Collect participant ID, visit number, number of blocks and check that the inputted variables are valid
exp_name = 'ZebraFliesTask_baseline'
exp_info = {
    'SubID': '',
    'Visit': '',
    'Blocks':'Ex. 4,6,8'}
while True:
    dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name)
    if dlg.OK == False:
        core.quit()
        sys.exit()

    # get blocks and write edf filename
    blocks = int(exp_info['Blocks'])
    participant_id = exp_info['SubID']
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
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"{participant_id}_{exp_name}_Visit{exp_info['Visit']}_{time_str}")
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

# Calculate host PC pixel conversions
host_x = int(scn_width//2)
host_y = int(scn_height//2)
px_per_cm = screen_w_px / screen_w_cm
px_per_dva = px_per_cm * (2 * view_dist_cm * math.tan(math.radians(0.5)))

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
current_qp = None # Setting global variable so we can print posteriors at the end

stim_domain = {'intensity': np.arange(0.01, 1, 0.01)}
param_domain = {
    'threshold': np.arange(0.01, 1, 0.01),
    'slope': 3.5,
    'lower_asymptote': 0.5, # Equal to chance
    'lapse_rate': np.arange(0, 0.05, 0.01) # Test 0:0.05 for adults, Consider 0:0.10 for children
}
outcome_domain = {'response': [1,0]}  # I'm going to flip this, to see if it fixes the way I intuitively think the algorithm should work; TDW 2025-01-22

# *TWO* QuestPlus staircases - one for each frequency condition
qp_high = QuestPlus(
    stim_domain=stim_domain,
    param_domain=param_domain,
    outcome_domain=outcome_domain,
    func='weibull',
    stim_scale='linear'
)

qp_low = QuestPlus(
    stim_domain=stim_domain,
    param_domain=param_domain,
    outcome_domain=outcome_domain,
    func='weibull',
    stim_scale='linear'
)

####### INITIALIZE EXPERIMENT TRIAL COMPONENTS #################################################################################################################################################################################################### 

kb = keyboard.Keyboard()
welcome_text = visual.TextStim(win=win, name='welcome_text',
    text='''Welcome to the Zebra Flies Game!''',
    font='Arial', units='deg', 
    pos=(0, 0), draggable=False, height=1.5, wrapWidth=1700, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',depth=0.0)
andy_text = visual.TextStim(win=win, text="This is Andy the Frog!", font='Arial', units='deg', pos=(0, 6), height=1.2, wrapWidth=1700, 
    color='black', colorSpace='rgb')
gabors_text = visual.TextStim(win=win, text="Andy loves to eat zebra flies like these!\n\n\n\n\n\n\n\n\n\n\n", 
    font='Arial', units='deg', pos=(0, 0), height=1.2, wrapWidth=1700, 
    color='black', colorSpace='rgb')
zebraflies_img = visual.ImageStim(win=win,
    image = "Images/zebraflies.png",
    name='zebraflies_img', units='deg', 
    mask=None, ori=0, pos=(0, -3), 
    size = (30,18.75), colorSpace='rgb')
instruct_text = visual.TextStim(win=win, name='instruct_text',
    text="", font='Arial', units='deg', pos=(0, 0), draggable=False, 
    height=1.2, wrapWidth=1700, ori=0, color='black', colorSpace='rgb', 
    opacity=1,languageStyle='LTR',depth=0.0)
gabor_inst1 = visual.GratingStim(
    win=win, name='gabor_inst1',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=90, pos=(-POSITION[0],POSITION[1]), size=(TARGET_SIZE, TARGET_SIZE), sf=(LOW_SPATIAL_FREQ), phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=0.5, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
gabor_inst2 = visual.GratingStim(
    win=win, name='gabor_inst2',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=0.0, pos=(POSITION[0],POSITION[1]), size=(TARGET_SIZE, TARGET_SIZE), sf=(HIGH_SPATIAL_FREQ), phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=0.5, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
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
gabor = visual.GratingStim(
    win=win, name='gabor',units='deg', 
    tex='sin', mask='gauss', anchor='center',
    ori=0.0, pos=(POSITION[0],POSITION[1]), size=(TARGET_SIZE, TARGET_SIZE), sf=(LOW_SPATIAL_FREQ), phase=0.0,
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

def drift_check():
    """ Performs a drift check. Allows for recalibration between blocks. """
    
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
            error = el_tracker.isRecording()
            if error == pylink.TRIAL_OK:
                print(error)
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

def show_end():
    ''' Displays Andy jumping and a thank you message. Calls terminate_task when esc is pressed.'''
    
    # Clear the psychopy window
    if win is not None:
        win.clearAutoDraw()
        win.flip()

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

def consecutive_check(trial_list):
    """ Checks the trial list to ensure that there are no more than MAX_CONSECUTIVE_TRIALS consecutive trials with the same sf condition """
    
    consecutive_count = 1
    if len(trial_list) == 0:
        return False
    for i in range(1, len(trial_list)):
        if trial_list[i]['freq_condition'] == trial_list[i - 1]['freq_condition']:
            consecutive_count += 1
            if consecutive_count > MAX_CONSECUTIVE_TRIALS:
                return False  # Invalid trial list
        else:
            consecutive_count = 1
    return True  # Valid trial list

# Get the full list of trials created by the handler
def create_trial_list(block_type):
    """ Create the trial list for the block. """
    
    # Get how many times to repeat each type of trial in TRIAL_TYPES
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
    """ Gets eye used. Returns 0 for left, 1 for right, None if eye data cannot be collected."""
    
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

def show_instructions(prac_block_num=None):
    """Display instructions based on block. """
    
    thisExp.addData('instructions.start', globalClock.getTime(format='float'))
        
    if prac_block_num in (1, 2, 3):
        instruct_text.text = f'''***PRACTICE LEVEL {prac_block_num}***\n\n
        Your job is to tell Andy which way the zebra flies are going!\n\n\n\n\n\n\n
        Press the left button if the zebra flies will move up and down, 
        and the right button if the zebra flies will move side to side.'''
        instruct_text.draw()
        andy_fix.draw()
        gabor_inst1.draw()
        gabor_inst2.draw()
        win.flip()
    else:
        instruct_text.text = '''Your job is to tell Andy which way the zebra flies are going!\n\n\n\n\n\n\n
        Press the left button if the zebra flies will move up and down, 
        and the right button if the zebra flies will move side to side.'''
        instruct_text.draw()
        andy_fix.draw()
        gabor_inst1.draw()
        gabor_inst2.draw()
        win.flip()

    keys = event.waitKeys(keyList=['space', 'escape','q'])
    if 'escape' in keys:
        terminate_task()
    elif 'q' in keys:
        show_end()
    elif 'space' in keys:
        thisExp.addData('instructions.end', globalClock.getTime(format='float'))
        return

def run_trial(trial, practice = False, practice_contrasts = None, block_num = None):
    """ Run one trial. Returns response: 1- Correct; 0 - Incorrent; None - No response. """
    
    # Reset variables
    t = 0
    frameN = -1
    routineTimer.reset()
    continueRoutine = True
    allKeys = []
    components = [fix_cross, andy_fix, gabor, kb]
    eye_used = None
    block_type = None
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
    
    # Set and target parameters 
    gabor.pos = np.array([POSITION[0] * trial['gabor_position'], POSITION[1]])
    gabor.ori = trial['orientation']
    frequency = HIGH_SPATIAL_FREQ if trial['freq_condition'] == 'High' else LOW_SPATIAL_FREQ
    gabor.sf = frequency
    
    # Set practice-specific variables
    if practice: 
        intensity = practice_contrasts.pop(0)
        gabor.contrast = intensity
        
        if block_num == 1:
            TARGET_DUR = None # target on screen for unlimited amount of time
            block_type = 'pract1'
        elif block_num == 2:
            TARGET_DUR = EXTENDED_TARGET_DUR
            block_type = 'pract2'
        else:
            TARGET_DUR = EXP_TARGET_DUR
            block_type = 'pract3'
            
    # Set QP algorithm logic for experiment trials
    else:
        TARGET_DUR = EXP_TARGET_DUR
        block_type = f'exp{block_num}'
        global current_qp
        if trial['freq_condition'] == 'High':
            current_qp = qp_high
        elif trial['freq_condition'] == 'Low':
            current_qp = qp_low
        
        # Save threshold, slope, and lapse rate from QP to data file
        threshold = current_qp.param_estimate['threshold']
        slope = current_qp.param_estimate['slope']
        lapse_rate = current_qp.param_estimate['lapse_rate']
        
        # Get next intensity from current staircase
        next_stim = current_qp.next_stim
        intensity = next_stim['intensity']
        
        # Update gabor contrast from QP
        gabor.contrast = intensity
        
    thisExp.addData('block', block_type)
    
    # Get trial duration
    if TARGET_DUR is not None:
        TRIAL_DUR = FIX_CROSS_DUR + ANDY_FIX_DUR + TARGET_DUR + RESPONSE_WINDOW
        thisExp.addData('trial_dur_s', TRIAL_DUR)
        
    # -------------------- Eyetracker Setup ---------------------------------
    # Esure tracker is ready to receive commands
    el_tracker = pylink.getEYELINK()
    el_tracker.setOfflineMode()
    el_tracker.sendCommand('clear_screen 0')
 
    # draw cross at fixation point on the host pc; params: x y size (pix)
    el_tracker.sendCommand(f'draw_cross {host_x} {host_y} 10')
        
    #draw cross at target locations
    dx = int(round(POSITION[0] * px_per_dva))
    el_tracker.sendCommand(f'draw_cross {host_x-dx} {host_y} 10')
    el_tracker.sendCommand(f'draw_cross {host_x+dx} {host_y} 10')
    
    # Print trial number on eyelink host monitor and output console
    if practice:
        print(f"Running practice trial {trial_index}:", trial)
        status_msg = 'PRACTICE TRIAL %d' % trial_index
        el_tracker.sendMessage('TRIALID %d' % trial_index)
    else:
        print(f"Running experiment trial {trial_index}:", trial)
        status_msg = 'TRIAL %d' % trial_index
        el_tracker.sendMessage('TRIALID %d' % trial_index)

    # Send status message to host PC
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)
    
    # Set trial variables
    el_tracker.sendMessage('!V TRIAL_VAR block %s' % block_type)
    el_tracker.sendMessage('!V TRIAL_VAR trial %s' % trial_index)
    el_tracker.sendMessage('!V TRIAL_VAR condition %s' % trial['freq_condition'])
    el_tracker.sendMessage('!V TRIAL_VAR gabor_pos %s' % trial['gabor_position'])
    el_tracker.sendMessage('!V TRIAL_VAR gabor_ori %s' % trial['orientation'])
    el_tracker.sendMessage('!V TRIAL_VAR gabor_intensity %s' % intensity)

    # put tracker in idle/offline mode before recording
    el_tracker.setOfflineMode()
    
    # Start recording
    try:
        el_tracker.startRecording(1, 1, 1, 1) # arguments: sample_to_file, events_to_file, sample_over_link, event_over_link (1-yes, 0-no)
    except RuntimeError as error:
        print("ERROR:", error)
        return 
    
    # Allocate time for the tracker to cache some samples
    pylink.pumpDelay(100) 
    
    # Get eye used. 
    eye_used = get_eye_used(el_tracker)
    if eye_used is None and not EYETRACKER_OFF:
        print(f"Could not get eye used on trial {trial_index}.")
        return
        
    # Return an error if tracker disconnected
    error = el_tracker.isRecording()
    if error is not pylink.TRIAL_OK:
        el_tracker.sendMessage('tracker_disconnected')
        print("Tracker disconnected.")
        return
    # ------------------------------------------------------------------------
    
    # Start trial while loop
    while continueRoutine: 
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1
        
        # Draw fixation cross at start of trial
        if fix_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            draw_comp(fix_cross, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('fixation_started')
            
        # Draw Andy fixation after fix cross duration; stays on until trial ends
        if andy_fix.status==NOT_STARTED and tThisFlip >= FIX_CROSS_DUR -frameTolerance:
            erase_comp(fix_cross, t, tThisFlipGlobal, frameN)
            draw_comp(andy_fix, t, tThisFlipGlobal, frameN)
            
        # Draw target and start checking for key presses
        if gabor.status == NOT_STARTED and tThisFlip >= FIX_CROSS_DUR+ANDY_FIX_DUR -frameTolerance:
            draw_comp(gabor, t, tThisFlipGlobal, frameN)
            el_tracker.sendMessage('target_started')
            draw_comp(kb, t, tThisFlipGlobal, frameN)
            win.callOnFlip(kb.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(kb.clearEvents, eventType='keyboard')
            
        if kb.status == STARTED: 
            key_name = None # None unless response is made
            rt = None # None unless response is made
            key = kb.getKeys(keyList=RESPONSE_KEYS, waitRelease=False)
            allKeys.extend(key)
            
            if allKeys: 
                last_key = allKeys[-1]
                key_name = last_key.name  # get just the last key pressed
                rt = last_key.rt
                continueRoutine = False # end trial when a key is pressed
                break
            
        if kb.getKeys(keyList=["escape"]):
            terminate_task()
        elif kb.getKeys(keyList=["q"]):
            show_end()
            
        # If statements when target duration is not unlimited
        if TARGET_DUR is not None:
            if gabor.status == STARTED and tThisFlipGlobal > gabor.tStartRefresh + TARGET_DUR -frameTolerance:
                erase_comp(gabor, t, tThisFlipGlobal, frameN)
                el_tracker.sendMessage('target_stopped')
                
            if tThisFlip >= TRIAL_DUR - frameTolerance: # end trial after trial duration has elapsed
                continueRoutine = False
                break
        
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
        if practice:
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
            current_qp.update(stim=next_stim, outcome={'response': response})
        elif practice:
            if response == 1:
                feedback_text.text = "Correct!"
                feedback_image.setImage("images/check_mark.png")
                feedback_image.draw()
                feedback_text.draw()
                happy_sound.play()
                win.flip()
                core.wait(FEEDBACK_DUR)
            else:
                feedback_text.text = "Try Again!"
                feedback_image.setImage("images/x_mark.png")
                feedback_image.draw()    
                feedback_text.draw()
                sad_sound.play()
                win.flip()
                core.wait(FEEDBACK_DUR)
    print("Response:", response, "RT:", rt)
    
    # Add trial data to the data file
    thisExp.addData('condition', trial['freq_condition'])
    thisExp.addData('gabor.intensity', intensity)
    thisExp.addData('gabor.sf', frequency)
    thisExp.addData('gabor.pos', 'L' if trial['gabor_position'] == -1 else 'R')
    thisExp.addData('gabor.ori', gabor.ori)
    thisExp.addData('keypress', key_name)
    thisExp.addData('accuracy', response)
    thisExp.addData('rt', rt)
    if not practice:
        thisExp.addData('qp_threshold', threshold)
        thisExp.addData('qp_slope', slope)
        thisExp.addData('qp_lapse', lapse_rate)
        print(f"Next Intensity: {intensity}")
    
    # Send trial data to EDF file
    try:
        el_tracker.sendMessage('!V TRIAL_VAR keypress %d' % key_name)
        el_tracker.sendMessage('!V TRIAL_VAR accuracy %d' % response)
    except:
        el_tracker.sendMessage('!V TRIAL_VAR rt -1')
    el_tracker.sendMessage('!V CLEAR 128 128 128')
    
    # Stop recording between trials to decrease size of output file
    el_tracker.stopRecording()
    pylink.pumpDelay(100) # add 100 msec to catch final events before stopping
    
    # Send trial result message to mark the end of the trial
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    thisExp.nextEntry() # Advance to the next row in the data file
        
    return response
    
def run_practice_block(block_num):
    """ Run all trials practice block. Must surpass accuracy threshold to move on. Experiment will terminate if accuracy threshold
        is not met within two tries. """
        
    show_instructions(block_num)
    
    accuracy = 0
    repeat_count = 0
    
    if block_num == 1:
        while accuracy <= ACCURACY_THRESHOLD and repeat_count < MAX_PRACTICE_REPEATS:
            drift_check()
            
            thisExp.addData(f'practice{block_num}.start', globalClock.getTime(format='float'))
            correct_count = 0
            repeat_count +=1 
            
            trial_list = create_trial_list('practice1')
            
            practice_contrasts = [PRACT1_CONTRASTS]*len(TRIAL_TYPES)
            
            for trial in trial_list:
                response = run_trial(trial, practice = True, practice_contrasts = practice_contrasts, block_num = block_num)
                if response is not None:
                    correct_count += response
                keys = event.waitKeys(keyList=['space', 'q'])
                if 'q' in keys:
                    show_end()
                
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
                    show_end()
                
            thisExp.addData(f'practice{block_num}.end', globalClock.getTime(format='float'))
    
    elif block_num > 1:
        while accuracy <= ACCURACY_THRESHOLD and repeat_count < MAX_PRACTICE_REPEATS:
            drift_check()
           
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
                    show_end()
                
            thisExp.addData(f'practice{block_num}.end', globalClock.getTime(format='float'))
        
    thisExp.nextEntry()

####### WELCOME SCREEN AND CALIBRATION  #################################################################################################################################################################################################### 

# Set clocks and start experiment
thisExp.status = STARTED
logging.setDefaultClock(globalClock)
win.flip()

# Window needs to be flipped before trying to run calibration or else code won't run
welcome_text.draw()
win.flip()
thisExp.addData('welcome.start', globalClock.getTime(format='float'))

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
    keys = event.waitKeys(keyList=['space', 'escape', 'q'])
    if 'escape' in keys:
        terminate_task()
    elif 'q' in keys:
        show_end()
    elif 'space' in keys:
        thisExp.addData('welcome.end', globalClock.getTime(format='float'))

####### INSTRUCTIONS  #################################################################################################################################################################################################### 

# Andy screen
andy_text.draw()
andy_fix.draw()
win.flip()
keys = event.waitKeys(keyList=['space', 'escape', 'q'])
if 'escape' in keys:
    terminate_task()
elif 'q' in keys:
    show_end()
    
# Gabors screen
gabors_text.draw()
zebraflies_img.draw()
win.flip()

keys = event.waitKeys(keyList=['space', 'escape', 'q'])
if 'escape' in keys:
    terminate_task()
elif 'q' in keys:
    show_end()

####### PRACTICE BLOCKS #################################################################################################################################################################################################### 

run_practice_block(1) # target stays on screen for unlimited amount of time, experimenter-paced
run_practice_block(2) # target presented for extended time
run_practice_block(3) # exactly like experiment trials
    
####### EXPERIMENT BLOCKS #################################################################################################################################################################################################### 

# Instruction text screen before experiment trials
show_instructions()

# Check drift before starting experiment
drift_check()

# Reset variables and generate the trial list
no_resp_trials = []
trial_list = create_trial_list('experiment')
block= 1

for trial in trial_list:
    response = run_trial(trial, practice = False, practice_contrasts = None, block_num = block)
    if response is None:
        no_resp_trials.append(trial)  

    # At every break interval, do a drift check to recalibrate if necessary
    if (trial['index'] % trials_per_block == 0 and trial['index'] != TOTAL_TRIALS) or (trial['index'] == TOTAL_TRIALS and len(no_resp_trials)>1): 
        
        break_text.draw()
        win.flip()
        block += 1
        print('\nNumber of trials to be repeated:', len(no_resp_trials), '\n') # print total number of trials with no response so far
        keys = event.waitKeys(keyList=['space', 'q'])
        if 'q' in keys:
            show_end()
        
        drift_check()

# Repeat trials with no response
trial_count = TOTAL_TRIALS
while len(no_resp_trials) > 0:
    block = 'rec'
    print(f"Re-running {len(no_resp_trials)} trials with no response...")

    remaining_trials = []

    for trial in no_resp_trials:
        trial_num = trial_list.index(trial) + 1
        thisExp.addData('trial', trial_num)
        trial_count += 1
        
        # At every break interval, do a drift check to recalibrate if necessary
        if trial_count % MAX_RECOVERY_TRIALS == 0: 
            
            break_text.draw()
            win.flip()
            print('\nNumber of trials to be repeated:', len(no_resp_trials), '\n') # print total number of trials with no response so far
            keys = event.waitKeys(keyList=['space', 'q'])
            if 'q' in keys:
                show_end()
            
            drift_check()

        response = run_trial(trial, practice = False, practice_contrasts = None, block_num = block)

        if response is None and trial['presented'] < MAX_TRIAL_REPEATS:
            remaining_trials.append(trial)

    no_resp_trials = remaining_trials

####### END EXPERIMENT #################################################################################################################################################################################################### 

show_end()