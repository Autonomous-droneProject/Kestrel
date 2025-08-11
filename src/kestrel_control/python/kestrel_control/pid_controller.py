import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration

class PIDController():
    #Clamping variables for frame rate (so it is not 0)
    MAX_TIME = Duration(nanoseconds = 100_000_000)
    MIN_TIME = Duration(nanoseconds = 1_000_000)

    #Clamping variables for integral
    MAX_I = 1.0
    MIN_I = -1.0

    def __init__ (self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.prev_time = self.clock.now()

        self.prev_error = 0.0

        self.i = 0.0

    def compute_command(self, offset): #or whatever name
        error = offset #error test, could be distance or motor offset

        curr_time = self.clock.now() # if dt needs to be taken out just delete this section and add to func def
        dt = curr_time-self.prev_time

        #Clamping duration bc max min do not understand ROS2
        if dt > self.MAX_TIME :
            dt = self.MAX_TIME
        elif dt < self.MIN_TIME :
            dt = self.MIN_TIME

        dt = dt.nanoseconds / 1e9

        p = self.kp * error #Calculate correction

        self.i += self.ki * error * dt #Handle over time error
        self.i = max(min(self.i, self.MAX_I), self.MIN_I) #Clamp I so not overshooting, picking out smallest num then biggest num
        
        derivative = (error - self.prev_error)/dt #Find derivative for d part
        d = self.kd * derivative #Handle overshoot

        pid = p + self.i + d #Put it all together

        self.prev_time = curr_time #Update prev_time
        self.prev_error = error #Set new previous error

        return pid
    
    #Just in case error should be reset
    def reset_error(self) :
        self.prev_error = 0.0
