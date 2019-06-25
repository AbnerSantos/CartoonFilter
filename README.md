# CartoonFilter
Filter that attempts to, given a photo or any other image, to "cartoonize" the image, similar to what toon and outline shaders do with 3D models, but applied to 2D.

# Final Report
We were able to reproduce the effect that toon shaders produce on 3D models but on 2D images instead, more specifically on cute pictures of cats and also furniture, which wasn't intially planned but worked out really great!

## Filters and Techniques Used
* Bilateral Filtering
* 2D Convolution with Laplacian Operator for Edge Detection
* Morphological Image Processing: Erosion and Dilation

## Libraries/Packages Used
* Python
* Numpy
* OpenCV

## Explanation
  Whilst searching for ways to get the result we wanted, we found references to several filters and techniques we could use, such as: Box blur filtering, Color image segmentation, Edge detection and more.
  At first we were thinking of segmenting the image by region growing and then coloring these regions with either the average or median of the colors in that region.
  Then, we thought of getting the hsv color space of the image and setting dynamic color thresholds or buckets to merge all similar colors into one, a histogram based filter.
  However, upon further research we came to the conclusion that the Bilateral Filtering method would be perfect for what we're trying to achieve, since it retains the edges while averaging the rest. It's generally used to remove noise, but if we turn it up to 11 it might as well give us the cartoonish feel we want!

First, we tried using the OpenCV bilateral filter, to test and see if the bilateral filter would produce what we wanted. 
After setting some variables, the OpenCV bilateral filter produced exactly what we were looking for: cartoonized cats!

![Sample Cat 1: OpenCV Bilateral Filter](Cats/Final/GatoFofinho_cartoon.png)

So, we decided to code our own Bilateral Filter, expecting we would get similar results, but that's not what happened.
Our Bilateral Filter didn't get to produce cartoonized cats, but stained glass cat filter!

![Sample Cat 1: Our Bilateral Filter](Cats/Final/GatoFofinho_stained_glass.png)

While making our bilateral filter, we faced a problem: the images were so big that the program would take too much time to run, sometimes taking more than 5 minutes. After that we decided to downsample the image, so the program could run faster.

<!-- Aqui colocam a explicacao do bilateral 
    Aumento da saturacao
    Mostra com mobília tambem
-->

We also decided to do some edge detection to create outlines for the images, but it would be optional to the user.
We start with a convolution with the Laplacian Operator ((-1, -1, -1), (-1, 8, -1), (-1, -1, -1)) as the kernel for the grayscale version of the cartoon image

![Sample Cat 1: Edge](Cats/Final/Edge.png)

Then applied a threshold:

![Sample Cat 1: Threshold Edge](Cats/Final/EdgeThreshold.png)

As we can see, there is still a lot of noise and it gets a lot of details we don't want.
To solve that, we decided to add some morphological operations to the edges. More especifically, an opening, witch is a dilation followed by an erosion.
After some testing we ended up using a circle-like mask/kernel of size 3 ((0, 1, 0), (1, 1, 1), (0, 1, 0)) for the and apply it only once, which led to great results:

![Sample Cat1: Treated Edges](Cats/Final/Erosion.png)

Great! A lot of the noise and unwanted details are gone.
We thought of then dilating to complete an opening operation, but we results weren't so great:

![Sample Cat1: Treated Edges 2](Cats/Final/Dilation.png)

We ended up not using it since the results were better without it. We also tried other possibilities, such as: 2 erosions, 1 dilation, 1 dilation 2 erosions, 2 erosions 2 dilations, and even applying a gaussian blur after the erosion, but overall, the best results were just to erode once.

Now we have to apply those edges to the image. Previously we would just add them in in white, because adding them in black led to highlights being lost, such as on the cat's eyes:

![Sample Cat1: Black Outlines](Cats/Final/Demon.png)

Yikes. Nope.

Then we decided for a while to leave the outlines white:

![Sample Cat1: White Outlines](Cats/Final/WhiteOutline.png)

Which is better, but still not optimal.
So we tried a different approach: In drawings, the outlines aren't always black or white, generally they are just a darker/brighter, more saturated version of the color in that area. So, we decided that if the brightness of a pixel in the outline is high, we decrease it by 20%, or else if it's low, we increase it by 20%:

![Sample Cat1: Final Outlines](Cats/Final/FinalOutline.png)

Not too shabby! It's not what we wanted to achieve initially but it does a great job of making a good looking outline whilst sharpening the edges as well.
