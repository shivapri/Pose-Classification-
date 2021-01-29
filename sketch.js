
let video;
let poseNet;
let pose;
let skeleton;
let brain;
let state='waiting';
let targetLabel;
function keyPressed(){
  if(key=='s'){
brain.saveData();
  }
  else{
  targetLabel = key;
  console.log(targetLabel);
  setTimeout(function(){
    console.log('collecting')
     state='collecting';
    setTimeout(function(){
    console.log('not collecting');
      state='waiting';
    },10000);
    
},10000);
 
  }
}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
  
  
  let options = {
  inputs:34,
  outputs:3,
  task:'classification',
    debug:true
    
  }
  
  brain = ml5.neuralNetwork(options);
  // brain.loadData('postures.json',dataReady);
  const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
  }
  brain.load(modelDetails,poseLoaded);
}
function poseLoaded(){
  console.log('pose classification ready!');
  // brain.classify(inputs,gotResults)
  classifyPose();
}
function classifyPose(){
if(pose)
{
   let inputs=[];
   for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
     inputs.push(y);
    }
  brain.classify(inputs,gotResults);
}
  else
  {
setTimeout(classifyPose,100);
  }
}
function dataReady(){
  brain.normalizeData();
  brain.train({epochs:10},finished);
}
function finished(){
console.log('model trained');
  brain.save();
}
function gotPoses(poses) {
  //console.log(poses);
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  if(state=='collecting'){
  let inputs=[];
   for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
     inputs.push(y);
    }
  let target = [targetLabel];
  brain.addData(inputs,target);
  }
  }
  
}

function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  translate(video.width,0);
  scale(-1,1);
  image(video, 0, 0,video.width,video.height);

  if (pose) {
    let eyeR = pose.rightEye;
    let eyeL = pose.leftEye;
    let d = dist(eyeR.x, eyeR.y, eyeL.x, eyeL.y);
    fill(255, 0, 0);
    ellipse(pose.nose.x, pose.nose.y, d);
    fill(0, 0, 255);
    ellipse(pose.rightWrist.x, pose.rightWrist.y, 32);
    ellipse(pose.leftWrist.x, pose.leftWrist.y, 32);

    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0, 255, 0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }

    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(255);
      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
  }
}
function gotResults(error,results){
console.log(results);
  console.log(results[0].label);
  classifyPose();
}
