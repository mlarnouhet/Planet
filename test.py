from dm_control import suite
from dm_control.suite.wrappers import pixels
import numpy as np
import cv2

env = suite.load(domain_name="cartpole", task_name="swingup", visualize_reward=True)
env = pixels.Wrapper(env)
spec = env.action_spec()
print(spec)
print(env.observation_spec())

time_step = env.reset()
while not time_step.last():
    obs = time_step.observation["pixels"]

    cv2.imshow("Video", obs)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)

cv2.destroyAllWindows()

print("Done !")