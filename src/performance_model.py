
def qr1_gpu(M,N,h2d_b,d2h_b,f_qr):
    #print(M)
    t_h2d = (M*N*8)/(h2d_b*(1024**3))
    t_qr  = (2*M*N**2 - (2/3)* N**3)/(f_qr*(1000**3))
    t_d2h = (N**2)/(d2h_b*(1024**3))
    t_total = t_h2d + t_qr + t_d2h
    print("qr1 time (s) : %f" %t_total)
    return t_total


def qr2_cpu(M,N,f_qr):
    t_qr  = (2*M*N**2 - (2/3)* N**3)/(f_qr*(1000**3))
    print("qr2 time (s): %f" %t_qr)
    return

def mm_gpu(M,N,h2d_b,d2h_b,f_mm):
    t_h2d   = (N*N*8)/(h2d_b*(1024**3))
    t_mm    = (M*N*(2*N-1))/(f_mm*(1000**3))
    t_d2h   = (M*N)/(d2h_b*(1024**3))
    t_total = t_mm + t_d2h
    print("mm time (s): %f" %t_total)
    return
    
qr1_gpu(1000/4,   100  ,2.13   ,1.12  , 2.34   )
qr1_gpu(10000/4,  100  ,4.77   ,3.95  , 12.22  )
qr1_gpu(100000/4, 100  ,4.71   ,6.08  , 22.20 )
qr1_gpu(1000000/4,100  ,4.57   ,2.86  , 34.86 )

qr2_cpu(400,   100  ,2.86)
#qr2_cpu(10000/4,  100  ,4.77   ,3.95  , 12.22  )
#qr2_cpu(100000/4, 100  ,4.71   ,6.08  , 22.20 )
#qr2_cpu(1000000/4,100  ,4.57   ,2.86  , 34.86 )


mm_gpu(1000/4,   100  ,2.13  , 2.4 , 28.38   )
mm_gpu(10000/4,  100  ,4.85  , 4.9 , 160.31  )
mm_gpu(100000/4, 100  ,4.98  , 6.6 , 209.65  )
mm_gpu(1000000/4,100  ,4.21  , 2.9 , 225.91  )



