# Bouncing Balls 2D

For these animations, the jpgs were later created in to gifs.

02-ChatGPT (with help):

<img src="https://github.com/Kinvert/Cuda-Stuff/blob/master/Simulations/Bouncing-Balls/2D/02-ChatGPT-Result.gif" width="480" height="360"/>

03-ChatGPT (with help):

<img src="https://github.com/Kinvert/Cuda-Stuff/blob/master/Simulations/Bouncing-Balls/2D/03-ChatGPT-Result.gif" width="360" height="480"/>

04-gave up on ChatGPT:

<img src="https://github.com/Kinvert/Cuda-Stuff/blob/master/Simulations/Bouncing-Balls/2D/04-Result.gif" width="480" height="360"/>

## Explanation

ChatGPT did not do well with this. Maybe the physics is a bit beyond what it was able to memorize from other people's writing.

This was tough enough I did need to go back and review one of my Engineering textbooks.

Statics and Dynamics 11th - R.C. Hibbeler [Check Price on Amazon](https://amzn.to/3GfJINv)

If you buy through that link I'll get a small percentage of Amazon's profit and you won't pay a penny more.

The description of how to do this is on page 245 of the Dynamics half of the book (the book is around 1,300 pages, this is around page 850 but the numbers start over for Dynamics since this is 2 books in 1)

There is more homework similar to this on page 254.

I had forgotten the equation for Coefficient of Restitution so I referenced the book.

Basically you just apply Conservation of Momentum and the Coefficient of Restitution to get the updated velocity vectors.

By ChatGPT's attempt 03 I gave up and wrote 04. That's where the working code starts.
