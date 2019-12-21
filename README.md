## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/evanlohn/106aProj/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.



## Introduction
Have you ever wanted your resident large, expensive humanoid robot to help you solve a children's puzzle? We aim to show how a Baxter robot can be programmed to solve a jigsaw puzzle with computer vision techniques and minimal human interaction. More concretely, we give the robot an image of the completed puzzle, along with the puzzle's dimensions and number of pieces, then place pieces on the table one by one and have Baxter place them with correct position and orientation.

## Design + Implementation

### Hardware
We worked with:
- 1 arm of the [Baxter robot](https://en.wikipedia.org/wiki/Baxter_(robot)) with [suction gripper kit](https://www.generationrobots.com/en/401622-vacuum-gripper-starter-kit-for-baxter.html)
- large floor puzzle: 24 pieces
- Logitech C920 webcam + tripod
- [AR tag](https://en.wikipedia.org/wiki/ARTag)

### Calibration

We break up our approach to puzzle solving into two main stages: calibration and a piece solver loop. 

The purpose of the calibration stage is to connect the pixel coordinates of images from the camera with physical coordinates usable by baxter. To do this, we:

1. Place a square AR tag at the upper left corner of the area in which Baxter will solve the puzzle
2. Capture an image from the webcam, segment the AR tag, and calculate a deskewing transformation using the expected orientation and aspect ratio (1:1 for a square) of the AR tag when viewed top-down. This deskewing transformation allows us to transform raw images from the camera into a "deskewed pixel space" that simulates what the image would like like if the camera were directly above the table, pointing downwards. By calculating the center of the segmented AR tag in deskewed pixel space, we get the origin of our puzzle-solving coordinate frame in deskewed pixel space.
3. Place baxter's camera directly above the AR tag pointing down. We then process the images from Baxter's hand camera with AR tag recognition software  to find the coordinate transformation between Baxter's hand and the AR tag. This allows us to calculate the coordinate transformation between Baxter and the middle of the AR tag. We store this transformation, allowing Baxter to know where the origin of the puzzle-solving coordinate frame is even after the AR tag is removed.
4. Remove the AR tag. Baxter is now ready to solve the puzzle!

### Physical Piece Solving

Physical Piece solving (picking up and placing a new piece with roughly correct position and orientation) uses the following loop:
1. Capture an image of the table before the new piece is placed
2. Put the new piece on the table, at the origin and capture another image. Using pixel-wise absolute difference between this image and the one from the previous step (and some additional processing), we segment the new puzzle piece

## Challenges

As can be seen from our videos, one of the main challenges we faced was the losses of precision incurred in the conversion from pixel position to physical position. The main causes of these losses were problems with the baxter hand camera
mention calibration issues, precision issues, segmentation issues

### Results

### Conclusions


This website showcases the work of Rebecca Abraham, Jason Huynh, and Evan Lohn on a primarily self-directed EECS C106A class project.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/evanlohn/106aProj/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
