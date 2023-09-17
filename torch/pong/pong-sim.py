# a simple pong game simulation that can run headless or can be extended with a GUI

import math
import random
import time

class PongObject:
    """
    PongObject is the base class for all objects in the game
    """

    def __init__(self, game):

        self.game = game

    def tick(self, t):
        pass


class Ball(PongObject):
    """
    Ball is the class for the ball object
    """

    def __init__(self, game):
        super().__init__(game)

        self.x = 0
        self.y = 0

        self.radius = 0.1

        self.vx = 0
        self.vy = 0

        self.ax = 0
        self.ay = 0

    def tick(self, t):
        pass


    
class Obstacle(PongObject):
    """
    Obstacle is a base class for all objects that are not affected by physics
    """

    def __init__(self, game):
        super().__init__(game)

        self.x = 0
        self.y = 0

        self.width = 0
        self.height = 0

    def tick(self, t):
        pass

    def checkCollision(self, ball):
        pass

    def handleCollision(self, ball):
        pass


class Paddle(Obstacle):
    """
    Paddle is the class for the paddle object
    """

    def __init__(self, game):
        super().__init__(game)

        self.width = 0.04
        self.height = 0.2

    def tick(self, t):
        pass

    def checkCollision(self, ball):
        pass

    def handleCollision(self, ball):
        pass


    

class Pong:

    def __init__(self):
        self.TPS = 60

        self.time = 0

        self.height = 1
        self.width = 2

        self.ball = Ball(self)
        self.paddle1 = Paddle(self)
        self.paddle2 = Paddle(self)

        self.objects = [self.ball, self.paddle1, self.paddle2]


    def tick(self):
        self.time += 1/self.TPS

        for obj in self.objects:
            obj.tick(self.time)
            
    
    def reset(self):
        pass

