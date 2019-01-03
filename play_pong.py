import pygame, sys
from pygame.locals import *
import time
import numpy as np
import pickle, h5py
import random
import pylab
#from mlp import Model
from player import Player
# Number of frames per second
# Change this value to speed up or slow down your game
FPS = 500

#Global Variables to be used through our program
#DRAW = False
ll = Player(6,9,0.0002,memory_pool_size = 100000,exp_epoch=3000,epsilon_rate = 0.0004,filename = sys.argv[1])
WINDOWWIDTH = 400
WINDOWHEIGHT = 300
LINETHICKNESS = 10
PADDLESIZE = 75
PADDLEOFFSET = 20

# Set up the colours
BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)

#Draws the arena the game will be played in.
def drawArena():
    if True:
        DISPLAYSURF.fill((0,0,0))
        #Draw outline of arena
        pygame.draw.rect(DISPLAYSURF, WHITE, ((0,0),(WINDOWWIDTH,WINDOWHEIGHT)), LINETHICKNESS*2)
        #Draw centre line
        pygame.draw.line(DISPLAYSURF, WHITE, (int(WINDOWWIDTH/2),0),(int(WINDOWWIDTH/2),int(WINDOWHEIGHT)), int(LINETHICKNESS/4))


#Draws the paddle
def drawPaddle(paddle):
    #Stops paddle moving too low
    if paddle.bottom > WINDOWHEIGHT - LINETHICKNESS:
        paddle.bottom = WINDOWHEIGHT - LINETHICKNESS
    #Stops paddle moving too high
    elif paddle.top < LINETHICKNESS:
        paddle.top = LINETHICKNESS
    #Draws paddle
    if True:
        pygame.draw.rect(DISPLAYSURF, WHITE, paddle)


#draws the ball
def drawBall(ball):
    if True:
        pygame.draw.rect(DISPLAYSURF, WHITE, ball)

#moves the ball returns new position
def moveBall(ball, ballDirX, ballDirY):
    ball.x += ballDirX
    ball.y += ballDirY
    return ball

#Checks for a collision with a wall, and 'bounces' ball off it.
#Returns new direction
def checkEdgeCollision(ball, ballDirX, ballDirY):
    if ball.top == (LINETHICKNESS) or ball.bottom == (WINDOWHEIGHT - LINETHICKNESS):
        ballDirY = ballDirY * -1
    if ball.left == (LINETHICKNESS) or ball.right == (WINDOWWIDTH - LINETHICKNESS):
        ballDirX = ballDirX * -1
    return ballDirX, ballDirY
def checkHitBall(ball, paddle1, paddle2, ballDirX):
    if ballDirX == -1 and paddle1.right >= ball.left and paddle1.left <= ball.left and paddle1.top <= ball.top and paddle1.bottom >= ball.bottom:
        return -1
    elif ballDirX == 1 and paddle2.right >= ball.right and paddle2.left <= ball.right and paddle2.top <= ball.top and paddle2.bottom >= ball.bottom:
        return -1
    else: return 1

#Checks to see if a point has been scored returns new score
def checkPointScored(paddle1, ball, score, ballDirX):
    #reset points if left wall is hit
    if ball.left == LINETHICKNESS:
        return -1
    #1 point for hitting the ball
    elif ballDirX == -1 and paddle1.right >= ball.left and paddle1.left <= ball.left and paddle1.top <= ball.top and paddle1.bottom >= ball.bottom:
        score += 1
        return score
    #5 points for beating the other paddle
    elif ball.right == WINDOWWIDTH - LINETHICKNESS:
        print("5 points!")
        score += 5
        return score
    #if no points scored, return score unchanged
    else: return score

#Artificial Intelligence of computer player
def artificialIntelligence(ball, ballDirX, paddle2):
    #If ball is moving away from paddle, center bat
    if ballDirX == -1:
        if paddle2.centery < (WINDOWHEIGHT/2):
            paddle2.y += 1
        elif paddle2.centery > (WINDOWHEIGHT/2):
            paddle2.y -= 1
    #if ball moving towards bat, track its movement.
    elif ballDirX == 1:
        if paddle2.centery < ball.centery:
            paddle2.y += 1
        else:
            paddle2.y -=1
    return paddle2

#Displays the current score on the screen
def displayScore(score):
    resultSurf = BASICFONT.render('Score = %s' %(score), True, WHITE)
    resultRect = resultSurf.get_rect()
    resultRect.topleft = (WINDOWWIDTH - 150, 25)
    DISPLAYSURF.blit(resultSurf, resultRect)

def print_state(reward,paddle1,paddle2,ball):
    print("test:")
    print(reward)
    print(paddle1.y)
    print(paddle2.y)
    print(ball.x)
    print(ball.y)

def normalize(reward, paddle1,paddle2,ball,last_action):
    n_reward = float(reward)
    n_paddle1 = float((paddle1.y-150.0)/150.0)
    n_paddle2 = float((paddle2.y-150.0)/150.0)
    n_ballx = float((ball.x - 200.0)/200.0)
    n_bally = float((ball.y - 150.0)/150.0)
    n_last_action = float(last_action-1.0)
    return np.array([n_reward,n_paddle1,n_paddle2,n_ballx,n_bally,n_last_action])


def main():
    # a = np.array([])
    # b = np.array([])
    # pylab.plot(a,b, 'b')
    # pylab.show()
    print("initialize Learner ...")
    ##########################
    #q learnig parameters
    # train_interval = 100
    # tot_frame_counter = 0
    frame_counter = 0
    # buffer_counter = 0
    skip_frame = 10
    act = False
    # memory_pool_size = 10000
    # state_frame = 9
    # dlen = 6
    # state_length = dlen*state_frame
    # memory_pool = np.zeros([memory_pool_size,state_length])
    # model = Model(load_model(),gamma=0.99,learning_rate = 0.0001,freeze_interval=20)

    last_action = 0
    # epoch = 0
    # epsilon = 1.0
    # first_train = False
    # train_counter = 0
    #########################
    pygame.init()
    global DISPLAYSURF
    ##Font information
    global BASICFONT, BASICFONTSIZE
    BASICFONTSIZE = 20
    BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)

    if True:
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH,WINDOWHEIGHT))
        pygame.display.set_caption('Pong')

    #Initiate variable and set starting positions
    #any future changes made within rectangles
    ballX = WINDOWWIDTH/2 - LINETHICKNESS/2
    ballY = WINDOWHEIGHT/2 - LINETHICKNESS/2
    playerOnePosition = (WINDOWHEIGHT - PADDLESIZE) /2
    playerTwoPosition = (WINDOWHEIGHT - PADDLESIZE) /2
    score = 0
    reward = 0
    avg_reward = 0
    #Keeps track of ball direction
    ballDirX = -1 ## -1 = left 1 = right
    ballDirY = -1 ## -1 = up 1 = down

    #Creates Rectangles for ball and paddles.
    paddle1 = pygame.Rect(PADDLEOFFSET,playerOnePosition, LINETHICKNESS,PADDLESIZE)
    paddle2 = pygame.Rect(WINDOWWIDTH - PADDLEOFFSET - LINETHICKNESS, playerTwoPosition, LINETHICKNESS,PADDLESIZE)
    ball = pygame.Rect(ballX, ballY, LINETHICKNESS, LINETHICKNESS)

    #Draws the starting position of the Arena
    drawArena()

    drawPaddle(paddle1)
    drawPaddle(paddle2)
    drawBall(ball)

  #  pygame.mouse.set_visible(0) # make cursor invisible
    tot_reward = 0
    while True: #main game loop
        frame_counter += 1
        if frame_counter == skip_frame:
            frame_counter = 0
            act = True
        if True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            #mouse movement commands
            # elif event.type == MOUSEMOTION:
            #     mousex, mousey = event.pos
            #     paddle1.y = mousey

        drawArena()
        drawPaddle(paddle1)
        drawPaddle(paddle2)
        drawBall(ball)

        ball = moveBall(ball, ballDirX, ballDirY)
        ballDirX, ballDirY = checkEdgeCollision(ball, ballDirX, ballDirY)
        score = checkPointScored(paddle1, ball, score, ballDirX)
        if act:
            sys.stdout.flush()
            act = False
            reward = score
            score = 0
            s = normalize(reward,paddle1,paddle2,ball,last_action)
#            True = True
            last_action = ll.learn(s)
        if last_action == 1:
            paddle1.y -= 1
        elif last_action == 2:
            paddle1.y += 1

        ballDirX = ballDirX * checkHitBall(ball, paddle1, paddle2, ballDirX)
        paddle2 = artificialIntelligence (ball, ballDirX, paddle2)

        if True:
            displayScore(tot_reward)
            pygame.display.update()
            FPSCLOCK.tick(FPS)

if __name__=='__main__':
    main()
