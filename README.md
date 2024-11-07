# PyVideoFace 

A simple program that downloads a video from a URL, decomposes a video into frames, recognizes faces in those frames, and then crops the frame to only show the face. 

## Usage

Each URL that is passed into the program should be seperated by a space and the url should be enclosed in quotes.

`videoFace process {url}`

## TODO
- Implement depduping using https://github.com/idealo/imagededup
- Despagetify code
- remove excessive try-catch blocks
- Add a containerized runtime???
