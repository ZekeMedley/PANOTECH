#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Change this to change where the files are exported to. *)
SetDirectory["/Users/zekemedley/Desktop/projects/faces"]
imageCount = 0;
While[True,
	(* Wait so that we don't grab too many repeats. *)
	Pause[2];
	(* Extract faces from the webcam image. *)
	images = FindFaces[CurrentImage[], "Image", PerformanceGoal->"Speed"];
	(* Export each image to the current directory. *)
	For[i=1, i<=Length[images],i++,
		Export[ToString[imageCount]<>".jpg", images[[i]]];
		imageCount++	
	]
]
