## Testing the provided binary of Pic2Sym v1.0 (*Windows 64bits only*) ##

[Back to start page](../ReadMe.md)

It&#39;s *recommended* to read first the ***Conclusions*** from the bottom of the ***[examples page](results.md)***, to see *what to expect* and understand *what images and font types are more suitable for the transformation*.

Now the steps:

1. Choose a directory for this test (*TestDir*)
1. Download repository files to another folder.
1. Unpack **bin/Pic2Sym.zip** and **bin/dlls.zip** in *TestDir*. Copy there **bin/agpl-3.0.txt**, as well.
1. Copy the whole **res/** folder (*together with the folder*) to *TestDir*
1. Launch ***TestDir*/Pic2Sym.exe**. If it starts, jump to **7**. For errors about *missing dll\-s*, continue below
1. Non\-development machines might not have several *common dll-s*, so here are some links to **download them in _TestDir_**, if reported missing: [msvcp120.dll][], [msvcr120.dll][], [comdlg32.dll][] or [advapi32.dll][]. _Make sure the application starts before continuing to **7**_
1. *Optionally* install (*double\-clicking the file*) the *free* font ***BpMono bold*** from [here][1] or from  ***TestDir*/res/BpMonoBold.ttf** *to ensure your machine has at least one interesting font to work with*
1. Activate the main window and press **Ctrl\+P** to display the *Control Panel*.
1. Now some configurations for a *first* ***rapid*** transformation:
	- select a **small image** to transform
    - select a *font family* to use. Suggestion not to underestimate: use a **bold style** font family with **less than 400 symbols**
    - bring **[Structural Similarity][] slider to 0**
1. Finally hit **Transform**. At the end of the transformation you&#39;ll be able to *inspect the result* with the *Transparency slider* and the *Zoom feature* from the main window

--------
[Back to start page](../ReadMe.md)

[1]:http://www.dafont.com/bpmono.font
[msvcp120.dll]:http://files.dllworld.org/msvcp120.dll-12.0.21005.1-64bit_3075.zip
[msvcr120.dll]:http://files.dllworld.org/msvcr120.dll-12.0.21005.1-64bit_3122.zip
[comdlg32.dll]:http://files.dllworld.org/comdlg32.dll-6.1.7601.17514-64bit_181.zip
[advapi32.dll]:http://files.dllworld.org/advapi32.dll-6.3.9600.17031-64bit.zip
[Structural Similarity]:https://ece.uwaterloo.ca/~z70wang/research/ssim