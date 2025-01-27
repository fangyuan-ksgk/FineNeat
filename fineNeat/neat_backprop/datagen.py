import numpy as np

class DataGenerator:
    def __init__(self, train_size=200, test_size=200, noise_level=0.5, batch_size=10):
        self.train_size = train_size
        self.test_size = test_size
        self.noise_level = noise_level
        self.batch_size = batch_size
        
    def generate_xor_data(self, num_points=None, noise=None):
        num_points = num_points if num_points is not None else self.train_size
        noise = noise if noise is not None else self.noise_level
        
        data = []
        for _ in range(num_points):
            x = np.random.uniform(-5.0, 5.0) + np.random.normal(0, noise)
            y = np.random.uniform(-5.0, 5.0) + np.random.normal(0, noise)
            label = 1 if (x > 0 and y > 0) or (x < 0 and y < 0) else 0
            data.append((x, y, label))
        return np.array(data)

    def generate_spiral_data(self, num_points=None, noise=None):
        num_points = num_points if num_points is not None else self.train_size
        noise = noise if noise is not None else self.noise_level
        
        data = []
        n = num_points // 2

        def gen_spiral(delta_t, label):
            for i in range(n):
                r = i / n * 6.0
                t = 1.75 * i / n * 2 * np.pi + delta_t
                x = r * np.sin(t) + np.random.uniform(-1, 1) * noise
                y = r * np.cos(t) + np.random.uniform(-1, 1) * noise
                data.append((x, y, label))

        gen_spiral(0, 0)  # First spiral
        gen_spiral(np.pi, 1)  # Second spiral
        return np.array(data)

    def generate_gaussian_data(self, num_points=None, noise=None):
        num_points = num_points if num_points is not None else self.train_size
        noise = noise if noise is not None else self.noise_level
        
        data = []
        n = num_points // 2

        def gen_gaussian(xc, yc, label):
            for _ in range(n):
                x = np.random.normal(xc, noise * 1.0 + 1.0)
                y = np.random.normal(yc, noise * 1.0 + 1.0)
                data.append((x, y, label))

        gen_gaussian(2, 2, 1)  # Positive examples
        gen_gaussian(-2, -2, 0)  # Negative examples
        return np.array(data)

    def generate_circle_data(self, num_points=None, noise=None):
        num_points = num_points if num_points is not None else self.train_size
        noise = noise if noise is not None else self.noise_level
        radius = 5.0
        
        data = []
        n = num_points // 2

        def get_circle_label(x, y):
            return 1 if (x*x + y*y < (radius * 0.5)**2) else 0

        # Generate positive points inside the circle
        for _ in range(n):
            r = np.random.uniform(0, radius * 0.5)
            angle = np.random.uniform(0, 2 * np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noise_x = np.random.uniform(-radius, radius) * noise/3
            noise_y = np.random.uniform(-radius, radius) * noise/3
            label = get_circle_label(x, y)
            data.append((x + noise_x, y + noise_y, label))

        # Generate negative points outside the circle
        for _ in range(n):
            r = np.random.uniform(radius * 0.75, radius)
            angle = np.random.uniform(0, 2 * np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noise_x = np.random.uniform(-radius, radius) * noise/3
            noise_y = np.random.uniform(-radius, radius) * noise/3
            label = get_circle_label(x, y)
            data.append((x + noise_x, y + noise_y, label))

        return np.array(data)

    def generate_random_dataset(self, choice=None):
        if choice is None:
            choice = np.random.randint(0, 4)
            
        generators = {
            0: self.generate_circle_data,
            1: self.generate_xor_data,
            2: self.generate_gaussian_data,
            3: self.generate_spiral_data
        }
        
        generator = generators[choice]
        train_data = generator(self.train_size)
        test_data = generator(self.test_size)
        
        # Shuffle the data
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        
        return train_data, test_data

    def generate_batch(self, train_data):
        indices = np.random.randint(0, len(train_data), self.batch_size)
        return train_data[indices]
    