let walker;

function setup() {
	createCanvas(windowWidth, windowHeight);
	walker = new Walker();
	background("#F8F8FF");
}

function draw() {
	var r = random(255);
	var g = random(255);
	var b = random(255);
	strokeWeight(2);
	var stepsize = randomGaussian(10, 3)
	walker.step(stepsize);
	walker.render(r, g, b);
}

class Walker {
	constructor() {
		this.x = width/2;
		this.y = height/2;
	}

	render(r, g, b) {
		stroke(r, g, b);
		point(this.x, this.y);
	}

	step(stepsize) {
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