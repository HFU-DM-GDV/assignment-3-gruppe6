<a name="readme-top"></a>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This code creats a hybrid image by combining a lowpass and highpass filtered image. The resulting hybrid image is interprented differently at different viewing distances. 
This task is edited by Nico Berndt, Vadim Borejko and Melanie Müller.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Follow these steps to get the porject running on your computer.

### Prerequisites

This project needs Python 3.11.2 or higher to run.
Download and install Python 3.11.2 from https://www.python.org/downloads/


### Installation

1. Clone/download the repository
2. Make sure your pip is up-to-date
   ```sh
   python -m pip install --upgrade pip
   ```
3. Install dependencies
   ```sh
   python -m pip install .
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

To run the program, use the following command:
```sh
  python hybridImaging.py
```
On startup, the program loads two images from ./images/ and displays them on the left and right. 
Now you need to set the reference points for both images. To do this choose the reference points of both images, left click the desired points on the left image, then left click the desired points on the right image. Three points must be selected in each window for the transformation to be computed. After you placed in total 6 points (3 on each image), the transformed images is shown between both images.
Finally press 'c' on your keyboard to calculate the hybrid image using the left image as the highpass and right image as the lowpass. A new windows containing the hybrid image should show after it finished calculating. Press 'r' if you want to reset the transform- and hybrid- images and begin assigning new reference points. Press 'q' to quit the program.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Following python packages have been used in this program:

* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)
* [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
