
import numpy as np, numpy.linalg as npl, math, matplotlib.pyplot as plt

def spd_sqrt_and_invsqrt(M, tol=1e-12):
    w,V = npl.eigh(0.5*(M+M.T))
    if np.min(w) <= tol: raise ValueError("SPD fail")
    Ms=(V*np.sqrt(w))@V.T; Mi=(V*(1/np.sqrt(w)))@V.T
    return Ms,Mi

def givens_rotation(n,i,j,theta):
    Rm=np.eye(n); c,s=math.cos(theta),math.sin(theta)
    Rm[i,i]=c; Rm[j,j]=c; Rm[i,j]=-s; Rm[j,i]=s; return Rm

def H_gamma(P,R,G,gamma,p=1.0):
    Pinv=npl.inv(P); Rinv=npl.inv(R)
    M=Pinv + gamma*(p**2)*G@Rinv@G.T
    H=gamma*npl.inv(M); return 0.5*(H+H.T)

def build_policy(P,R,G,gamma,p=1.0):
    Bp = R + gamma*(p**2)*(G.T@P@G); Binv=npl.inv(Bp)
    def a_star(s,f): return -gamma*p*(Binv@(G.T@P@f(s)))
    return a_star

def ou_step(z,rho,sigma,rng):
    return rho*z + sigma*math.sqrt(max(1-rho**2,0.0))*rng.normal(0.0,1.0,size=z.shape)

def cropped_overlay(xy_opt, xy_zero, title, xlab, ylab, out_png, out_pdf, margin=0.15):
    x_opt,y_opt=xy_opt[:,0],xy_opt[:,1]
    xmin,xmax=float(np.min(x_opt)),float(np.max(x_opt))
    ymin,ymax=float(np.min(y_opt)),float(np.max(y_opt))
    dx,dy=max(xmax-xmin,1e-6),max(ymax-ymin,1e-6)
    xmin-=margin*dx; xmax+=margin*dx; ymin-=margin*dy; ymax+=margin*dy
    xz,yz=xy_zero[:,0],xy_zero[:,1]
    inside=(xz>=xmin)&(xz<=xmax)&(yz>=ymin)&(yz<=ymax)
    exit_idx=None
    for k in range(len(xz)):
        if not inside[k]: exit_idx=k; break
    plt.figure(dpi=220)
    plt.plot(x_opt,y_opt,label='Optimal a*(s)')
    if exit_idx is None: plt.plot(xz,yz,label='No control (a≡0)')
    else:
        if exit_idx>1: plt.plot(xz[:exit_idx], yz[:exit_idx], label='No control (a≡0)')
        ex=xz[exit_idx-1] if exit_idx>0 else xz[0]
        ey=yz[exit_idx-1] if exit_idx>0 else yz[0]
        plt.scatter([ex],[ey], marker='^', label='No-control exits frame')
    plt.scatter([x_opt[0]],[y_opt[0]], marker='x', label='Start')
    plt.scatter([x_opt[-1]],[y_opt[-1]], marker='o', label='Optimal end')
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    plt.axis('equal'); plt.xlim([xmin,xmax]); plt.ylim([ymin,ymax]); plt.legend()
    plt.savefig(out_png,bbox_inches='tight'); plt.savefig(out_pdf,bbox_inches='tight'); plt.show()

# ---------------- S1 Unicycle ----------------
dt,gamma=0.1,0.98
P=np.diag([1/9,1/9,1,1,1]); beta=0.63; Q=beta*P
R=np.diag([0.00390625,0.00390625])
G=np.zeros((5,2)); G[3,0]=dt; G[4,1]=dt
Sigma_white=np.diag([0.01**2,0.01**2,0.005**2,0.02**2,0.02**2])
alpha_xyvw,kappa_xyvw=0.28,1.2; alpha_vw,kappa_vw=0.25,1.5
def S1_of_s(s):
    phi,v,om=float(s[2]),float(s[3]),float(s[4])
    S=np.eye(5); S=givens_rotation(5,0,1,phi)@S
    ang_v=alpha_xyvw*math.tanh(kappa_xyvw*v); S=givens_rotation(5,0,3,ang_v)@S; S=givens_rotation(5,1,3,ang_v)@S
    nu=alpha_vw*math.tanh(kappa_vw*v*om); S=givens_rotation(5,3,4,nu)@S; return S
H=H_gamma(P,R,G,gamma); _,Hinv=spd_sqrt_and_invsqrt(H); PQs,_=spd_sqrt_and_invsqrt(P-Q)
f=lambda s: Hinv@(S1_of_s(np.asarray(s).reshape(-1))@(PQs@np.asarray(s).reshape(-1)))
a_star=build_policy(P,R,G,gamma,1.0)
tau_xy,sigma_xy=3.0,0.25; rho_xy=math.exp(-dt/tau_xy)
tau_vw,sigma_vw=1.5,0.12; rho_vw=math.exp(-dt/tau_vw)
D=np.zeros((5,4)); D[0,0]=1; D[1,1]=1; D[3,2]=1; D[4,3]=1
def rollout_S1(s0,T,policy='optimal',seed=7):
    rng=np.random.default_rng(seed); s=np.asarray(s0).reshape(-1)
    z_xy=rng.normal(0.0,sigma_xy,2); z_vw=rng.normal(0.0,sigma_vw,2)
    S_hist=[s.copy()]; A_hist=[]
    for _ in range(T):
        a=a_star(s,f) if policy=='optimal' else np.zeros(2); A_hist.append(a.copy())
        z_xy=ou_step(z_xy,rho_xy,sigma_xy,rng); z_vw=ou_step(z_vw,rho_vw,sigma_vw,rng)
        z_all=np.concatenate([z_xy,z_vw]); z=f(s)+(G@a)+(D@z_all)
        w=rng.normal(0.0,1.0,5)*np.sqrt(np.diag(Sigma_white)); s=z+w; S_hist.append(s.copy())
    return dict(s=np.array(S_hist), a=np.array(A_hist))
s0=np.array([3.0,-2.0,0.6,0.0,0.0]); T=300
ro=rollout_S1(s0,T,'optimal',7); rz=rollout_S1(s0,T,'zero',7)
xy_o,xy_z=ro["s"][:,[0,1]],rz["s"][:,[0,1]]
cropped_overlay(xy_o,xy_z,"S1 Unicycle (disturbed): no-control vs optimal","x [m]","y [m]","fig_S1_disturbed.png","fig_S1_disturbed.pdf")

# ---------------- S2 2-DOF manipulator ----------------
dt,gamma=0.05,0.99
P=np.diag([1/np.pi**2,1/np.pi**2,1,1]); beta=0.5265; Q=beta*P
R=np.diag([0.0009,0.0009]); G=np.zeros((4,2)); G[2,0]=dt; G[3,1]=dt
Sigma_white=np.diag([0,0,2e-4,2e-4])
alpha1,kappa1,beta0,kappa2,grav0=0.35,1.0,0.30,1.0,0.5
def S2_of_s(s):
    th1,th2,w1,w2=s; S=np.eye(4)
    ang1=alpha1*math.tanh(kappa1*th1*w1)+grav0*math.sin(th1)
    ang2=alpha1*math.tanh(kappa1*th2*w2)+grav0*math.sin(th2)
    ang12=beta0*math.tanh(kappa2*(th2-th1))
    S=givens_rotation(4,0,2,ang1)@S; S=givens_rotation(4,1,3,ang2)@S; S=givens_rotation(4,2,3,ang12)@S; return S
H=H_gamma(P,R,G,gamma); _,Hinv=spd_sqrt_and_invsqrt(H); PQs,_=spd_sqrt_and_invsqrt(P-Q)
f=lambda s: Hinv@(S2_of_s(np.asarray(s).reshape(-1))@(PQs@np.asarray(s).reshape(-1)))
a_star=build_policy(P,R,G,gamma,1.0)
tau_w,sigma_w=1.0,0.18; rho_w=math.exp(-dt/tau_w)
D=np.zeros((4,2)); D[2,0]=1; D[3,1]=1
def rollout_S2(s0,T,policy='optimal',seed=7):
    rng=np.random.default_rng(seed); s=np.asarray(s0).reshape(-1); z=rng.normal(0.0,sigma_w,2)
    S_hist=[s.copy()]; A_hist=[]
    for _ in range(T):
        a=a_star(s,f) if policy=='optimal' else np.zeros(2); A_hist.append(a.copy())
        z=ou_step(z,rho_w,sigma_w,rng); s=f(s)+(G@a)+(D@z)
        w=rng.normal(0.0,1.0,4)*np.sqrt(np.diag(Sigma_white)); s=s+w; S_hist.append(s.copy())
    return dict(s=np.array(S_hist), a=np.array(A_hist))
s0=np.array([1.2,-0.8,0.0,0.0]); T=320
ro=rollout_S2(s0,T,'optimal',7); rz=rollout_S2(s0,T,'zero',7)
xy_o,xy_z=ro["s"][:,[0,1]],rz["s"][:,[0,1]]
cropped_overlay(xy_o,xy_z,"S2 2-DOF Manipulator (disturbed): no-control vs optimal","θ1 [rad]","θ2 [rad]","fig_S2_disturbed.png","fig_S2_disturbed.pdf")

# ---------------- S3 point-mass ----------------
dt,gamma=0.08,0.985
P=np.diag([1/16,1/16,1,1]); beta=0.6; Q=beta*P
R=np.diag([0.012,0.012]); G=np.zeros((4,2)); G[2,0]=dt; G[3,1]=dt
Sigma_white=np.diag([0.01**2,0.01**2,0.02**2,0.02**2])
alpha_pv,kappa_pv=0.25,1.2; alpha_v,kappa_v=0.20,1.0
def S3_of_s(s):
    x,y,vx,vy=s; S=np.eye(4)
    ang=math.atan2(vy, vx+1e-6); S=givens_rotation(4,0,1,ang)@S
    angp=alpha_pv*math.tanh(kappa_pv*(abs(vx)+abs(vy)))
    S=givens_rotation(4,0,2,angp)@S; S=givens_rotation(4,1,3,angp)@S
    angv=alpha_v*math.tanh(kappa_v*(vx*vy)); S=givens_rotation(4,2,3,angv)@S; return S
H=H_gamma(P,R,G,gamma); _,Hinv=spd_sqrt_and_invsqrt(H); PQs,_=spd_sqrt_and_invsqrt(P-Q)
f=lambda s: Hinv@(S3_of_s(np.asarray(s).reshape(-1))@(PQs@np.asarray(s).reshape(-1)))
a_star=build_policy(P,R,G,gamma,1.0)
tau_xy,sigma_xy=2.5,0.20; rho_xy=math.exp(-dt/tau_xy)
tau_v,sigma_v=1.2,0.18; rho_v=math.exp(-dt/tau_v)
D=np.zeros((4,4)); D[0,0]=1; D[1,1]=1; D[2,2]=1; D[3,3]=1
def rollout_S3(s0,T,policy='optimal',seed=7):
    rng=np.random.default_rng(seed); s=np.asarray(s0).reshape(-1)
    z_xy=rng.normal(0.0,sigma_xy,2); z_v=rng.normal(0.0,sigma_v,2)
    S_hist=[s.copy()]; A_hist=[]
    for _ in range(T):
        a=a_star(s,f) if policy=='optimal' else np.zeros(2); A_hist.append(a.copy())
        z_xy=ou_step(z_xy,rho_xy,sigma_xy,rng); z_v=ou_step(z_v,rho_v,sigma_v,rng)
        z_all=np.concatenate([z_xy,z_v]); z=f(s)+(G@a)+(D@z_all)
        w=rng.normal(0.0,1.0,4)*np.sqrt(np.diag(Sigma_white)); s=z+w; S_hist.append(s.copy())
    return dict(s=np.array(S_hist), a=np.array(A_hist))
s0=np.array([4.0,-3.0,0.0,0.0]); T=280
ro=rollout_S3(s0,T,'optimal',7); rz=rollout_S3(s0,T,'zero',7)
xy_o,xy_z=ro["s"][:,[0,1]],rz["s"][:,[0,1]]
cropped_overlay(xy_o,xy_z,"S3 Point-Mass (disturbed): no-control vs optimal","x [m]","y [m]","fig_S3_disturbed.png","fig_S3_disturbed.pdf")

# ---------------- S4 3-DOF manipulator ----------------
dt,gamma=0.05,0.99
P=np.diag([1/np.pi**2,1/np.pi**2,1/np.pi**2,1,1,1]); beta=0.6; Q=beta*P
R=np.diag([0.0025,0.0025,0.0025]); G=np.zeros((6,3)); G[3,0]=dt; G[4,1]=dt; G[5,2]=dt
Sigma_white=np.diag([0,0,0, 2e-4,2e-4,2e-4])
alpha1,kappa1,grav0=0.3,1.0,0.6; beta12,beta23,kappaN=0.28,0.28,1.0
def S4_of_s(s):
    th1,th2,th3,w1,w2,w3=s; S=np.eye(6)
    ang1=alpha1*math.tanh(kappa1*th1*w1)+grav0*math.sin(th1)
    ang2=alpha1*math.tanh(kappa1*th2*w2)+grav0*math.sin(th2)
    ang3=alpha1*math.tanh(kappa1*th3*w3)+grav0*math.sin(th3)
    S=givens_rotation(6,0,3,ang1)@S; S=givens_rotation(6,1,4,ang2)@S; S=givens_rotation(6,2,5,ang3)@S
    a12=0.28*math.tanh(kappaN*(th2-th1)); a23=0.28*math.tanh(kappaN*(th3-th2))
    S=givens_rotation(6,3,4,a12)@S; S=givens_rotation(6,4,5,a23)@S; return S
H=H_gamma(P,R,G,gamma); _,Hinv=spd_sqrt_and_invsqrt(H); PQs,_=spd_sqrt_and_invsqrt(P-Q)
f=lambda s: Hinv@(S4_of_s(np.asarray(s).reshape(-1))@(PQs@np.asarray(s).reshape(-1)))
a_star=build_policy(P,R,G,gamma,1.0)
tau_w,sigma_w=1.2,0.16; rho_w=math.exp(-dt/tau_w)
D=np.zeros((6,3)); D[3,0]=1; D[4,1]=1; D[5,2]=1
def rollout_S4(s0,T,policy='optimal',seed=7):
    rng=np.random.default_rng(seed); s=np.asarray(s0).reshape(-1); z=rng.normal(0.0,sigma_w,3)
    S_hist=[s.copy()]; A_hist=[]
    for _ in range(T):
        a=a_star(s,f) if policy=='optimal' else np.zeros(3); A_hist.append(a.copy())
        z=ou_step(z,rho_w,sigma_w,rng); s=f(s)+(G@a)+(D@z)
        w=rng.normal(0.0,1.0,6)*np.sqrt(np.diag(Sigma_white)); s=s+w; S_hist.append(s.copy())
    return dict(s=np.array(S_hist), a=np.array(A_hist))
s0=np.array([1.0,-0.6,0.4,0.0,0.0,0.0]); T=340
ro=rollout_S4(s0,T,'optimal',7); rz=rollout_S4(s0,T,'zero',7)
xy_o,xy_z=ro["s"][:,[0,1]],rz["s"][:,[0,1]]
cropped_overlay(xy_o,xy_z,"S4 3-DOF Manipulator (disturbed): no-control vs optimal","θ1 [rad]","θ2 [rad]","fig_S4_disturbed.png","fig_S4_disturbed.pdf")

print("Saved: fig_S1_disturbed.png/pdf, fig_S2_disturbed.png/pdf, fig_S3_disturbed.png/pdf, fig_S4_disturbed.png/pdf")
