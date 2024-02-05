# code from https://www.youtube.com/watch?v=c3tX_qReGIM﻿
# https://www.glowscript.org/#/user/maxwell.fazio/folder/SCWIResourceFolder/program/SimpeHarmonicOscillator/edit

from vpython import *  # 7.6.2
g = 9.81  # 중력가속도 9.81 m/s^2

# 화면 객체를 생성합니다. backgroun 색과 카메라 center 포인트를 설정한다
scene = canvas(background = vector(0.36, 0.47, 0.23), center = vector(0, -0.7, -0.5))

# 천장의 물체를 생성합니다. 재질은 wood이고 색깔은 orange입니다.
ceiling = box(length = 1, width = 1, height = 0.01, color = color.orange)

# ball을 생성합니다. 
# 반지름은 0.1m
# 초기 위치는 y방향으로 -0.5m입니다. (vpython에서는 y방향이 밑방향입니다)
# 질량은 0.5kg
# 초기속도는 0입니다
ball = sphere(radius = 0.1, color = color.green, opacity=0.8)
ball.pos = vector(0, -0.5, 0)
ball.m = 20
ball.v = vector(0,0,0)

# 스프링을 생성합니다. 코일 수는 15번, radius(반지름)와 thickness(두께)를 설정합니다 
# 길이는 천장에서 ball까지로 설정합니다
spring = helix(coils = 15, radius = 0.05, thickness = 0.01)
spring.pos = ceiling.pos
spring.axis = ball.pos - spring.pos

# spring.L에는 초기 ball의 위치값이 들어갑니다. (여기서는 0.5m로 고정됩니다)
# 스프링상수는 10 N/m입니다
spring.L = abs(ball.pos.y)
spring.k = 2

# 텍스트를 표시하는 라벨객체를 생성합니다
label1 = label()
label2 = label()
label3 = label()
 
# 스프링의 힘을 리턴하는 함수
def spring_F(spring):
    return -spring.k * (spring.length - spring.L) * spring.axis.norm()

#------------------------------------------------------------------
# 애니메이션 코드 
#------------------------------------------------------------------
 
dt = 0.01
t = 0

while True:
    rate(100)           # [Hz] 만큼 루프를 지연시킵니다
 
    F = spring_F(spring)      # 힘을 구해서 F 변수에 저장
    
    ball.a = vector(0, -g, 0) + F / ball.m   # 가속도
    ball.v += ball.a * dt                    # 속도
    ball.pos += ball.v  * dt                 # 위치
 
    spring.axis = ball.pos - spring.pos   # 스프링이 공을 따라가게 하기 위한 코드
 
    t += dt
 
    # 위치, 속도, 가속도를 표시합니다
    label1.pos = ceiling.pos + vector(0.5,0.2,0)
    label1.text = ('position is : %1.2f' % ball.pos.y)
    label2.pos = ceiling.pos + vector(0.5,0.05,0)
    label2.text = ('velocity is : %1.2f' % ball.v.y)
    label3.pos = ceiling.pos + vector(0.5,-0.10,0)
    label3.text = ('acc is : %1.2f' % ball.a.y)