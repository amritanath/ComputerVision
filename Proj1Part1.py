import cv2
import numpy as np
import sys


if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

tmp = np.copy(inputImage)
myListB=[]
myListG=[]
myListR=[]
for i in range(H1, H2) :
    for j in range(W1, W2) :
        b, g, r = inputImage[i, j]
       # Converting R8 G8 B8 divide by 255 and apply inverse gamma to get Linear RGB       
        #       Blue
        b1 = float(b/255)
        g1 = float(g/255)
        r1 = float(r/255)
        if b1 < 0.03928:
            myListB.append(b1/12.92)
        else :
            myListB.append( pow  (((b1 + 0.055)/1.055), 2.4))
 
        #       Green
        
        if g1 < 0.03928:
            myListG.append(g1/12.92)
        else :
            myListG.append( pow  (((g1 + 0.055)/1.055), 2.4))

        #       Red
        
        if r1 < 0.03928:
            myListR.append(r1/12.92)
        else :
            myListR.append( pow  (((r1 + 0.055)/1.055), 2.4))

#print(len(myListB))
#print(len(myListG))
#print(len(myListR))            

#print("= STAGE 3 OUTPUT")
#for i in range( len(myListB)) :
#    print(" r = " , myListR[i])
#    print(" g = " , myListG[i])
#    print(" b = " , myListB[i])
#    print("====================")

# Converting linearRGB To XYZ

a = [[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]]

X =[]
Y =[]
Z =[]
for i in range(len(myListB)) :
    d =[float(myListR[i]), float(myListG[i]), float(myListB[i])]
    e = np.dot(a, d)
    X.append(e[0])
    Y.append(e[1])
    Z.append(e[2])

#for i in range(len(X)) :
#    print("X = ", X[i])
#    print("Y = ", Y[i])
#    print("Z = ", Z[i])
#    print("**********")

# Converting XYZ TO L to find min max of L

t=[]


for i in range(len(Y)) :
    t.append(float(Y[i]/ 1.0))


L=[]

for i in range(len(t)) :
    if t[i]>0.008856 :
        L.append(   float((116*(pow(t[i] ,(1/3)))) - 16))
    else :
        L.append( float(903.3 * t[i]))

#print(len(L))

min = float(100.00)
max = float(0.00)

for i in range(len(L)) :
    if L[i] > max :
        max = float(L[i])
    if L[i] < min :
        min = float(L[i])

#print("min")
#print(min)
#print("max")
#print(max)

#cv2.imshow('tmp', tmp)

# end of example of going over window

#outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

myListB[:]=[]
myListG[:]=[]
myListR[:]=[]

for i in range(0, rows) :
    for j in range(0, cols) :
        b, g, r = inputImage[i, j]
        # Converting R8 G8 B8 divide by 255 and apply inverse gamma to get Linear RGB
        #       Blue
        b1 = float(b/255)
        g1 = float(g/255)
        r1 = float(r/255)
        if b1 < 0.03928:
            myListB.append(b1/12.92)
        else :
            myListB.append( pow  (((b1 + 0.055)/1.055), 2.4))
        
        #       Green
        
        if g1 < 0.03928:
            myListG.append(g1/12.92)
        else :
            myListG.append( pow  (((g1 + 0.055)/1.055), 2.4))
        
        #       Red
        
        if r1 < 0.03928:
            myListR.append(r1/12.92)
        else :
            myListR.append( pow  (((r1 + 0.055)/1.055), 2.4))


# Converting linearRGB To XYZ

a = [[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]]

X[:] =[]
Y[:] =[]
Z[:] =[]
for i in range(len(myListB)) :
    d =[float(myListR[i]), float(myListG[i]), float(myListB[i])]
    e = np.dot(a, d)
    X.append(e[0])
    Y.append(e[1])
    Z.append(e[2])

#print (len(X))
#print (len(Y))
#print (len(Z))



uw = (4 * 0.95) /( 0.95 + ( 15* 1.0) + (3*1.09) )
vw = (9 * 1.0)  /( 0.95 + ( 15* 1.0) + (3*1.09) )

t[:]=[]
u1 =[]
v1 =[]

for i in range(len(Y)) :
    t.append(float(Y[i]/ 1.0))


L[:]=[]

for i in range(len(t)) :
    if t[i]>0.008856 :
        L.append(   float((116*(pow(t[i] ,(1/3)))) - 16))
    else :
        L.append( float(903.3 * t[i]))


# print("L LENGTH:", len(L))

# calculating  U V


for i in range(len(X)) :
    d = float(X[i] +( 15.0 * Y[i]) +( 3.0 * Z[i]))
    if float(d) == 0.0 :
        u1.append(0.0)
        v1.append(0.0)
    else :    
        u1.append( float((4.0 * X[i]) / d))
        v1.append( float((9.0 * Y[i]) / d))

u = []
v = []

for i in range(len(X)) :
    k = float((13 * L[i] * 1.0) * ( u1[i] - uw))
    k= round(k,5)
    if  k == 0.0 :
        u.append(0.0)
        v.append(0.0)
    else :
        u.append(float(k))
        v.append(float(k))


#for i in range(len(t)) :
#    print("L :", L[i])
#    print("u: ", u[i])
#    print("v: ", v[i])
#    print("%%%%%%%%%%%%%%%%%%%")

# calculating L1 = (L[i]- min/(max-min))* 100 (linear strechin )


L1 =[]
for i in range(len(L)) :
    if L[i]>= min and L[i] <=max :
        L1.append( float(((L[i]- min)/(max-min))* 100.0))
    elif L[i] < min :
        L1.append(0.0)
    else :
        L1.append(100.0)
#    print(L1[i])
#    print(":::::::::::::::")

#DOUBT 
#print(len(L1))
#print(len(u))
#print(len(v))

# converting L1uv  to XYZ

u2 = []
v2 = []

for i in range(len(L)) :
    if L1[i] == 0.0 :
        u2.append(0.0)
        v2.append(0.0)
    else :    
        u2.append(float((u[i]+( 13 * uw * L1[i]))/(13*L1[i])))
        v2.append(float((v[i]+( 13 * vw * L1[i]))/(13*L1[i])))

    


X2 = []
Y2 = []
Z2 = []

for i in range(len(L1)) :
    if L1[i] > 7.9996 :
        Y2.append( round(pow( ((L1[i]+16)/116), 3) * 1.0,5))
    else :
        Y2.append(float((L1[i]/903.3) *1.0))

for i in range(len(Y2)) :
    if v2[i] == 0.0:
        X2.append(0.0)
        Z2.append(0.0)
    else :    
        X2.append( Y2[i] * 2.25 * ( u2[i]/v2[i] ))
        Z2.append(  round(    (Y2[i] * (3 - (0.75 * u2[i])-  (5*v2[i]) ) / v2[i]),5))

    

#converting X2Y2Z2 to linear RGB
    
a2=[[3.240479, -1.53715, -0.498535], [-0.969256, 1.875991, 0.041556], [0.055648,-0.204043, 1.057311]]


Rs = []
Gs = []
Bs = []
#CHECK IF I CAN MAKE VALUES LESS THAN 0.000001 =0 TO AVOID NAN
for i in range(len(X2)) :
    d1 =[float(X2[i]), float(Y2[i]), float(Z2[i])]
    e1 = np.dot(a2, d1)
    if(e1[0] < 0):
        Rs.append(0.0)
    elif(e1[0] >1):
        Rs.append(1.0)
    else:
        Rs.append(float(e1[0]))
    if(e1[1] < 0):
        Gs.append(0.0)
    elif(e1[1] >1):
        Gs.append(1.0)
    else:
        Gs.append(float(e1[1]))
    if(e1[2] < 0):
        Bs.append(0.0)
    elif(e1[2] >1):
        Bs.append(1.0)
    else:
        Bs.append(float(e1[2]))

#for i in range(len(Bs)):
#    if(Bs[i] < 0 or Bs[i] >1 or Gs[i] <0 or Gs[i] >1 or Rs[i] <0 or Rs[i] >1):
#        print('problem')

R8 =[]
G8 =[]
B8 =[]

for i in range (len(Rs)):
    if(Rs[i] < 0.00304):
        R8.append (float(12.92 * Rs[i]))
    else:
        R8.append(float(   (1.055 * (pow(Rs[i],(1/2.4))))-0.055))

    if(Gs[i] < 0.00304):
        G8.append (float(12.92 * Gs[i]))
    else:
        G8.append(float(   (1.055 * (pow(Gs[i],(1/2.4))))-0.055))

    if(Bs[i] < 0.00304):
        B8.append (float(12.92 * Bs[i]))
    else:
        B8.append(float(   (1.055 * (pow(Bs[i],(1/2.4))))-0.055))



finalB = []
finalG = []
finalR = []

for i in range(len(B8)) :  
    finalB.append(int(round(B8[i] * 255)))
    finalG.append(int(round(G8[i] * 255)))
    finalR.append(int(round(R8[i] * 255)))

#for i in range(len(finalB)) :
#    
#    print("B =", finalB[i])
#    print("G =", finalG[i])
#    print("R =", finalR[i])
#    print(i)

k =0
outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)
#print(" ========")
for i in range(0, rows) :
    for j in range(0, cols) :
#        b, g, r = inputImage[i, j]
        outputImage[i,j] = [finalB[k], finalG[k], finalR[k]]
        k=k+1
                            
cv2.imshow("output:", outputImage)
cv2.imwrite(name_output, outputImage);



# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
