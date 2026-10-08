These are two images of will smith eating spaghetti, one is straight from a feaver dream and the other one looks very real.
Between those two images are 2 years of research and particle physics. This area of generative models has been moving insanely fast recently, today i will walk you through the advancements made in diffusion and flow matching.

The core goal of this field is to generate very complex data like images.
Lets consider images: within all possible combinations of pixels, a few combinations of those are more likely to actually resemble a meaningful image.
Lets look at a simplified version of that: this plot shows a distribution across two axis, the bumbs here show actual images, as we move away from those peaks the images become more random and less resembling of an actual image.

If we were able to map out this distribution we could easily generate images by just sampling. As you can imagine this distribution consists of 1024x1024= ... dimensions .. so unfathomably complex. We really need to find a better way.

# Diffusion
Lets start with picking any point in our distribution. As you see once we just pick any point we are no longer interested in mapping out the entire distribution, now we can only care about the direction towards the next peak. Starting anywhere gives us noise, thus our problem turns into turning a noisy image into a real one.

The motion we are seeing here is
