import time

class PIDController():
    #CENTER = (0, 0) For PD servos finding distance, may try to implement
    #Clamping variables for frame rate (so it is not 0)
    MAX_TIME = 0.1
    MIN_TIME = 0.001
    #Clamping variables for integral
    MAX_I = 1
    MIN_I = -1

    def __init__ (self, kp, ki, kd):
        #If you are using a servo, just use PD, should be ki=0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # Center marking for servo pd, I think open cv has a funky grid which is why this was 
        # included but could be sorted by bridge file
        # self.centerX = 0 May replace the center pos with a tuple/coordinates
        # self.centerY = 0
        # self.center = CENTER
        self.prev_error = 0.0
        self.prev_time = 0.0
        self.curr_time = time.perf_counter() #I gotta review this section
        self.dt = self.curr_time

    def step(self, offset):
        error = offset #error test, could be distance or motor offset
        self.prev_time = self.curr_time #prev time is time of last step
        self.curr_time = time.perf_counter() #time counter find new frame rate
        dt = max(min(self.curr_time - self.prev_time, self.MAX_TIME), self.MIN_TIME) #Finding time change and clamping it
        p = self.kp * error #Calculate correction
        i += self.ki * error * dt #Handle over time error
        i = max(min(i, self.MAX_I), self.MIN_I) #Clamp I so not overshooting, picking out smallest num then biggest num
        derivative = (error - self.prev_error)/dt #Find derivative for d part
        d = self.kd * derivative #Handle overshoot
        pid = p + i + d #Put it all together
        self.prev_error = error #Set new previous error
        return pid
    
    #Just in case error should be reset
    def reset_error(self) :
        self.prevError = 0
