# Matlab_LQR

This folder contains a MATLAB LQR demonstration for a wheel-legged cart with a parallel five-bar leg visualization.

## Model Scope

The controller uses a course-level equivalent model:

- The plant is modeled as a wheel-driven inverted pendulum.
- The state is `x = [position; velocity; body_pitch; body_pitch_rate]`.
- The input is the horizontal wheel-ground force.
- The parallel five-bar mechanism is visualized kinematically to show wheel, body, hip joints, knee joints, and the two closed-chain leg paths.

This is suitable for automatic-control coursework and LQR explanation. It is not a full rigid-body multibody derivation of the five-bar linkage.

## How to Run

Open MATLAB, then run:

```matlab
cd Matlab_Wheeled_legged
run_wheel_leg_lqr_demo
```

The demo will:

1. Build the linearized state-space model.
2. Compute the LQR gain with `lqr(A, B, Q, R)`.
3. Simulate nonlinear cart-pendulum dynamics with saturated control.
4. Animate the wheel-legged cart and five-bar leg structure.
5. Plot position, pitch angle, and control input histories.

## Files

- `run_wheel_leg_lqr_demo.m`: main script.
- `wheel_leg_lqr_config.m`: physical, control, simulation, and visualization parameters.
- `linearized_wheel_leg_model.m`: small-angle state-space model for LQR design.
- `wheel_leg_dynamics.m`: nonlinear simulation model with LQR feedback.
- `animate_wheel_leg_lqr.m`: animation of the wheel-legged robot.
- `five_bar_leg_points.m`: display geometry for the parallel five-bar linkage.
- `draw_rotated_rectangle.m`: body drawing helper.
- `plot_response.m`: response curves.
- `saturate.m`: scalar saturation helper.

## Notes for Reports

The design logic is:

1. Around the upright equilibrium, the wheel-legged vehicle is approximated as a wheel-driven inverted pendulum.
2. The linear model is used only for the LQR gain calculation.
3. The closed-loop response is tested on a nonlinear pendulum model.
4. The five-bar leg is drawn as a mechanism-level visualization so the control result is easier to interpret.
