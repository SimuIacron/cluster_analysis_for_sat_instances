import os
import winsound


# makes a beeping noise, useful to tell user that experiment has finished
def make_noise():
    if os.name == 'nt':
        duration = 500
        winsound.Beep(440, duration)
        winsound.Beep(587, duration)
        winsound.Beep(440, duration)