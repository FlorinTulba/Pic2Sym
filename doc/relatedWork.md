## Related work ##
[Back to start page](../ReadMe.md)

Most applications in this area first average the brightness of every cell to be approximated within the original image. Then they quantize those means, resulting only a few brightness bins, instead of 255. Finally they associate an appropriately large ASCII character to each bin so that the symbol&#39;s size will let it be perceived as the desired luma level.

Generating color results is typically tackled by generating chroma using colored symbols (their foreground) and again, the size of the glyphs standing for luma.

A **remarkably short Python solution** for an image\-to\-ASCII grayscale transformer with 24 symbols, black background and text output is [here](https://gist.github.com/cdiener/10491632).

**Quite a different approach** on image\-to\-ASCII is demonstrated on [this page](https://larc.unt.edu/ian/art/ascii/color/). It uses 4 or 3 colored neighbor symbols to represent a single pixel from the original image.

Another interesting related application is **VLC**, which has an [ASCII play mode](https://www.youtube.com/watch?v=fuQjDfZ9lV4) running in a console with its 16 available colors.

Other applications **don&#39;t stick to using same size symbols** or to **position them vertically**.

----------

Among the many image/video\-to\-ASCII applications, only a few tackle:

1.	***large symbol sets*** (more difficult to find appropriate matches quick enough)
2.	the problems raised by the ***less\-ideal aspect\-ratio of the original symbols*** (results are less accurate vertically, as symbols occupy vertical rectangles, so vertical information will be less frequently approximated than horizontal image data)
3.	using a ***non\-black (or white) background*** for the symbols (results appear like a rough canvas). [This Android application](https://play.google.com/store/apps/details?id=com.muri.asciiart&hl=en) addressed the issue.

**Pic2Sym** attempts to handle the issues above. Point 1 involves developing fast and accurate match algorithms. One such algorithm (accurate, but slow) I included in the application comes from [Structural Similarity](https://ece.uwaterloo.ca/~z70wang/research/ssim) research. Point 2 meant resizing the symbols to fit within squares of desired size. Point 3 simply averages the color around the chosen symbol to determine its background.


----------

[Back to start page](../ReadMe.md)

