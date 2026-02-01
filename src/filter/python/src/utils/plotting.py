import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def xyPlot(title, labelx, labely, 
    vec1, label1, 
    vec2 = None, label2 = None, 
    vec3 = None, label3 = None,
    vec4 = None, label4 = None):
    plt.plot(vec1[:, 0], vec1[:, 1], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:, 0], vec2[:, 1], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:, 0], vec3[:, 1], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:, 0], vec4[:, 1], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)


def xyztPlot(title, 
    vec1, label1, 
    vec2 = None, label2 = None,
    vec3 = None, label3 = None,
    vec4 = None, label4 = None):
    plt.subplot(311)
    plt.plot(vec1[:,0], vec1[:,1], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:,0], vec2[:,1], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:,0], vec3[:,1], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:,0], vec4[:,1], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel('t(s)')
    plt.ylabel('x(m)')
    plt.title(title)

    plt.subplot(312)
    plt.plot(vec1[:,0], vec1[:,2], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:,0], vec2[:,2], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:,0], vec3[:,2], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:,0], vec4[:,2], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel('t(s)')
    plt.ylabel('y(m)')

    plt.subplot(313)
    plt.plot(vec1[:,0], vec1[:,3], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:,0], vec2[:,3], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:,0], vec3[:,3], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:,0], vec4[:,3], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel('t(s)')
    plt.ylabel('z(m)')


def plotBiases(ts, bg, ba):
    fig = plt.figure('IMU biases')
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.subplot(211)
    plt.plot(ts, bg[:,0], label="x")
    plt.plot(ts, bg[:,1], label="y")
    plt.plot(ts, bg[:,2], label="z")
    plt.grid()
    plt.legend()
    plt.title('Gyro bias')
    plt.xlabel('t')
    plt.ylabel('bias [rad/s]')

    plt.subplot(212)
    plt.plot(ts, ba[:,0], label="x")
    plt.plot(ts, ba[:,1], label="y")
    plt.plot(ts, ba[:,2], label="z")
    plt.grid()
    plt.legend()
    plt.title('Accel bias')
    plt.xlabel('t')
    plt.ylabel('bias [m/s2]')


def make_position_plots(traj, gt):
    # 2d positions
    fig = plt.figure('2D views')
    gs = gridspec.GridSpec(2, 2)
    
    fig.add_subplot(gs[:, 0])
    xyPlot('XY plot', 'x(m)', 'y(m)', 
        traj[:, 1:3], 'estim. traj', 
        gt[:, 1:3], 'gt')
    
    fig.add_subplot(gs[0, 1])
    xyPlot('XZ plot', 'x(m)', 'z(m)', 
        traj[:, [1,3]], 'estim. traj', 
        gt[:, [1,3]], 'gt')

    fig.add_subplot(gs[1, 1])
    xyPlot('YZ plot', 'y(m)', 'z(m)', 
        traj[:, [2,3]], 'estim. traj', 
        gt[:, [2,3]], 'gt')
    fig.subplots_adjust(hspace=0.4)


def make_velocity_plots(est_vel, gt_vel):
    plt.figure("Velocity")
    
    plt.subplot(311)
    plt.plot(est_vel[:,0], est_vel[:,1], label='est')
    plt.plot(gt_vel[:,0], gt_vel[:,1], label='gt')
    plt.title('Velocity')
    plt.xlabel('t(s)')
    plt.ylabel('x(m/s)')
    plt.legend()
    plt.grid()

    plt.subplot(312)
    plt.plot(est_vel[:,0], est_vel[:,2], label='est')
    plt.plot(gt_vel[:,0], gt_vel[:,2], label='gt')
    plt.xlabel('t(s)')
    plt.ylabel('y(m/s)')
    plt.legend()
    plt.grid()

    plt.subplot(313)
    plt.plot(est_vel[:,0], est_vel[:,3], label='est')
    plt.plot(gt_vel[:,0], gt_vel[:,3], label='gt')
    plt.xlabel('t(s)')
    plt.ylabel('z(m/s)')
    plt.legend()
    plt.grid()


def make_ori_euler_plots(est_xyz, gt_xyz):
    plt.figure("Orientation [Euler angles]")

    plt.subplot(311)
    plt.plot(est_xyz[:, 0], est_xyz[:, 1], label='est')
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 1], label='gt')
    plt.title('Attitude')
    plt.xlabel('t(s)')
    plt.ylabel('yaw(deg)')
    plt.legend()
    plt.grid()

    plt.subplot(312)
    plt.plot(est_xyz[:, 0], est_xyz[:, 2], label='est')
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label='gt')
    plt.xlabel('t(s)')
    plt.ylabel('pitch(deg)')
    plt.legend()
    plt.grid()

    plt.subplot(313)
    plt.plot(est_xyz[:, 0], est_xyz[:, 3], label='est')
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 3], label='gt')
    plt.xlabel('t(s)')
    plt.ylabel('roll(deg)')
    plt.legend()
    plt.grid()

