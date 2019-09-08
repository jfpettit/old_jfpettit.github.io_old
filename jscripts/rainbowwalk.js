let walker;

function setup() {
	createCanvas(windowWidth, windowHeight);
	walker = new Walker();
	background("#F8F8FF");
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight)
}

function draw() {
	var r = random(255);
	var g = random(255);
	var b = random(255);
	strokeWeight(3);
	var stepsize = randomGaussian(10, 3)
	walker.step(stepsize);
	walker.render(r, g, b);
}

class Walker {
	constructor() {
		this.x = randomGaussian(width/2, width/4);
		this.y = randomGaussian(height/2, height/4);
		this.prevx = this.x;
		this.prevy = this.y;
	}

	render(r, g, b) {
		//line(this.x, this.y, this.prevx, this.prevy);
		stroke(r, g, b);
		point(this.x, this.y);
	}

	step(stepsize) {
		this.prevx = this.x;
		this.prevy = this.y;
		var choice = floor(random(4));
		if (choice === 0) {
			this.x+=stepsize;
		}
		else if (choice == 1) {
			this.x-=stepsize;
		}
		else if (choice == 2) {
			this.y+=stepsize;
		}
		else if (choice == 3) {
			this.y-=stepsize;
		}
		this.x = constrain(this.x, 0, windowWidth-1);
		this.y = constrain(this.y, 0, windowHeight-1)
	}
}