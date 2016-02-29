## Example results ##
[Back to start page](../ReadMe.md)

The first version of the application has a sequential match algorithm and constitutes a tool for experimenting with various match aspects. It contains some optimizations, but next versions will focus more on speed.

Below are some results, which can also be found in [doc/examples](examples/) folder. The mentioned directory contains the originals as well. Each result file is prefixed by the name of the processed image.

Clicking on the presented cases will open the full\-size results from [doc/examples](examples/).

| Satisfactory&nbsp;results&nbsp;and&nbsp;their&nbsp;settings&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Comments | Duration |
|:-|:-|:-|
| <br>Default settings; 27540 patches to approximate using 125 symbols (*BPmono_Bold_APPLE_ROMAN_10*)<br>[![](Example1.jpg)](examples/6_BPmono_Bold_APPLE_ROMAN_10_2.5_0.25_1_0.75_0.1_0.2_0.2_0.1_0_2040_1350.jpg) | It marks high in the following parts: eyes, eyebrows, wrinkles, some contours and texture of the shiny blue object in the close plane.<br>The grid splits incur some unnatural transitions.<br><br>The bold symbols of the Apple Roman encoding of [BpMono](http://www.dafont.com/bpmono.font) font family usually produce pretty results. (The font is free and provided also in the [res](../res/) folder. To be visible to Pic2Sym, *it needs to be installed*.) | 153 s |
| <br>Default settings; 27540 patches to approximate using 220 symbols (*Envy_Code_R_Regular_APPLE_ROMAN_10*)<br>[![](Example2.jpg)](examples/6_Envy Code R_Regular_APPLE_ROMAN_10_2.5_0.25_1_0.75_0.1_0.2_0.2_0.1_0_2040_1350.jpg) | The necklace, the guitar tail and some of the eyes look still ok.<br>The aspect is quite blurry, as a consequence of not using a bold font.<br><br>Some of the glyphs used here are quite similar, in the sense they have just different accents. Such differences have almost no impact on the result, except the longer time required generating it. | 256 s |
| <br>Default settings; 27540 patches to approximate using 201 symbols (*ProFontWindows_Regular_APPLE_ROMAN_10*)<br>[![](Example3.jpg)](examples/6_ProFontWindows_Regular_APPLE_ROMAN_10_2.5_0.25_1_0.75_0.1_0.2_0.2_0.1_0_2040_1350.jpg) | Approximated eyebrows, eyes, face contours and hairs have a fair\-enough appearance.<br><br>Although declared as *Regular*, the symbols look bold\-ish. | 240 s |
| <br>Default settings; 27540 patches to approximate using 214 symbols (*Anonymous_Pro_Bold_APPLE_ROMAN_10*)<br>[![](Example4.jpg)](examples/13_Anonymous Pro_Bold_APPLE_ROMAN_10_2.5_0.25_1_0.75_0.1_0.2_0.2_0.1_0_2040_1350.jpg) | Objects thinner than the font size normally can&#39;t maintain their aspect.<br>Their background decides how clear they remain. | 250 s |
| <br>Default settings, but with [Structural Similarity][] disabled; 27405 patches to approximate using 191 symbols (*BPmono_Bold_UNICODE_10*)<br>[![](Example5.jpg)](examples/1_BPmono_Bold_UNICODE_10_0_0.25_1_0.75_0.1_0.2_0.2_0.1_0_2030_1350.jpg) | The thin lines on the quasi\-uniform wall are well approximated.<br>Besides that, disabling [Structural Similarity][] produced the result several times faster. However, the method shouldn&#39;t be underestimated. | 30 s |
| <br>Using only [Structural Similarity][], this time; 27405 patches to approximate using 191 symbols (*BPmono_Bold_UNICODE_10*)<br>[![](Example6.jpg)](examples/1_BPmono_Bold_UNICODE_10_2.5_0_0_0_0_0_0_0_0_2030_1350.jpg) | [Structural Similarity][] took more than 5 times the duration required by all the other techniques. Still, it captures additional subtleties when comparing the letters on the board and many more. | 172 s |

- - -

| Less&nbsp;satisfactory&nbsp;results&nbsp;and&nbsp;their&nbsp;settings&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Comments | Duration |
|:-|:-|:-|
| <br>Using only [Structural Similarity][], again; 27540 patches to approximate using 341 symbols (*Monaco_Regular_UNICODE_10*)<br>[![](Example7.jpg)](examples/15_Monaco_Regular_UNICODE_10_2.5_0_0_0_0_0_0_0_0_2040_1350.jpg) | The chamois seem quite blurred and the background seriously competes for viewer&#39;s attention, more than it should, in my opinion. | 385 s |
| <br>Still using only [Structural Similarity][]; 27405 patches to approximate using 219 symbols (*Consolas_Italic_APPLE_ROMAN_10*)<br>[![](Example8.jpg)](examples/7g_Consolas_Italic_APPLE_ROMAN_10_2.5_0_0_0_0_0_0_0_0_1350_2030.jpg) | *Italic* fonts cannot tackle top\-left corners well\-enough.<br>There are just a few parts that looks like one would expect from an approximation.<br>A reason might be that there are many large background differences among neighbor patches. | 186 s |
| <br>Finally, default settings, but with [Structural Similarity][] disabled; 625 patches to approximate using 218 symbols (*Courier_New_Bold Italic_APPLE_ROMAN_10*)<br>[![](Example9.jpg)](examples/17g_Courier_New_Bold_Italic_APPLE_ROMAN_10_0_0.25_1_0.75_0.1_0.2_0.2_0.1_0_250_250.jpg) | ***Bold Italic*** fonts don&#39;t serve well when patches contain vertical texture, like the pajama of the man from top\-right corner. | 1 s |

- - -

The speed of generating the approximations depends on the machine running the application, so the enumerated durations are just orientative.

For better understanding how to configure the application, read the [Control Panel](CtrlPanel.md) material.


##### Several conclusions:
Such transformations are better suitable when:
- the images have *large\-enough uniform regions* and *clear contours* (***more clarity***)
- the *irrelevant parts from the scene are sufficiently dark and/or uniform* (***less focus shift***)
- the *symbols approximating the image* are:
	* *bold* (approximation is ***more than a grid of blurred cells***)
	* *of a small\-enough size* (***better accuracy***)

Cases to avoid:
- pictures with *lots of random context changes between patches* (they generate ***odd mosaics***)
- *symbol sets* which:
	* contain more than 400 glyphs (***incurred time costs; little perceived difference on result***)
	* have a *size much too small* to distinguish them
	* have *lots of almost identical glyphs* (***large time\-penalty for little quality improvement***)
	* are *italic* when there are many regions with textures tilted differently
	* contain *glyphs filling almost solid their square* (such symbols are very likely to approximate ambiguous patches and ***they appear rigid, despite they are correctly selected***)

----------
[Back to start page](../ReadMe.md)

[Structural Similarity]:https://ece.uwaterloo.ca/~z70wang/research/ssim