import paddle
import math


class SimpleGrda:
    def __init__(self, params, lr=0.01, c=0.0, mu=0.7):
        self.params = [x for x in params]
        self.lr = lr
        self.c = c
        self.mu = mu
        self.iterations = 0
        # self.accumulators = [paddle.to_tensor(x.detach().cpu().numpy())
        #                      for x in self.params]
        self.accumulators = [paddle.create_parameter(x.shape, x.dtype,
                                                     default_initializer=paddle.nn.initializer.Uniform(-0.1, 0.1))
                             for x in self.params]
        self.l1_accumulation = 0

    def step(self):
        c = self.c
        mu = self.mu
        lr = self.lr
        l1_diff = c * math.pow(lr, (0.5 + mu)) * math.pow(self.iterations + 1., mu) - c * math.pow(lr, (
                    0.5 + mu)) * math.pow(self.iterations + 0., mu)
        self.l1_accumulation += l1_diff
        first_iter = max(1 - self.iterations, 0)

        updates = []
        grads = [x.grad for x in self.params]

        for p, g, a in zip(self.params, grads, self.accumulators):
            new_a = a + first_iter * p - self.lr * g
            updates.append((a, new_a))
            new_a_l1 = paddle.abs(new_a) - self.l1_accumulation
            new_p = paddle.sign(new_a) * paddle.clip(new_a_l1, min=0)
            updates.append([p, new_p])

        for raw_value, new_value in updates:
            raw_value.set_value(new_value)

        self.iterations += 1

    def clear_grad(self):
        for p in self.params:
            if not p.stop_gradient:
                p.clear_gradient()
