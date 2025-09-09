# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 11:25:05 2023

@author: Edward Wardell s2072334
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class GluonField:
    def __init__(self, lattice_size):
        
        # Initialises an empty field with a given size
        self.lattice_size = lattice_size
        
        # Links will make a 2D array that will contain the SU(2) matrix which will describe
        # The field at each link of the lattice.
        # One matrix will be required for either direction of the lattice
        self.horizontal_links = np.zeros((lattice_size, lattice_size, 2, 2), dtype = np.complex128)
        self.vertical_links = np.zeros((lattice_size, lattice_size, 2, 2), dtype = np.complex128)

    def set_horizontal_link(self, i, j, su2):
        
        # Sets the value for the field at a specific site (i, j)
        self.horizontal_links[i, j] = np.conj(su2)
        
    def set_vertical_link(self, i, j, su2):
        
        self.vertical_links[i, j] = np.conj(su2)
        
    def recall_horizontal_link(self, i, j):
        
        # Gets back the values for an su2 at a specific link
        return self.horizontal_links[i, j]
    
    def recall_vertical_link(self, i, j):
        
        return self.vertical_links[i, j]
    
def random_su2_matrix():
    
        spread =  0.4
       
        s1 = np.matrix([[0,1],[1,0]])
        s2 = np.matrix([[0,-1j],[1j,0]])
        s3 = np.matrix([[1,0],[0,-1]])  
       
        a = np.random.uniform(-0.5, 0.5)
        b = np.random.uniform(-0.5, 0.5)
        c = np.random.uniform(-0.5, 0.5)
        d = np.random.uniform(-0.5, 0.5)

        norm = np.sqrt(b**2 + c**2 + d**2)
       
        x_0 = np.sign(a) * np.sqrt(1-(spread)**2)
        x_1 = (b*spread)/norm
        x_2 = (c*spread)/norm
        x_3 = (d*spread)/norm
       
        x_0 = np.sign(a) * np.sqrt(1-(spread)**2)
       

        #su2 = np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]])
        random_su2 = np.identity(2)*x_0 + (s1*x_1 + s2*x_2 + s3*x_3) * 1j
       
        return random_su2

class Lattice:
    
    def __init__(self, lattice_size):
        self.lattice_size = lattice_size
        
        # Represents a lattice where the gluon field is defined, stores a specific gluon field config on lattice.
        self.field = GluonField(lattice_size)
        
    def calculate_plaquette(self, i, j):
        horizontal_link_1 = self.field.recall_horizontal_link(i, j)
        horizontal_link_2 = self.field.recall_horizontal_link(i, (j + 1) % self.lattice_size)
        vertical_link_1 = self.field.recall_vertical_link(i, j)
        vertical_link_2 = self.field.recall_vertical_link((i + 1) % self.lattice_size, j)

        plaquette = horizontal_link_1 @ horizontal_link_2 @ np.conj(vertical_link_2) @ np.conj(vertical_link_1)

        return plaquette
        
    def calculate_action(self):
        action = 0.0
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                plaquette_1 = self.calculate_plaquette(i, j)
                plaquette_2 = self.calculate_plaquette(i, (j + 1) % self.lattice_size)

        action += 1 - np.real(np.trace(plaquette_1))
        action += 1 - np.real(np.trace(plaquette_2))

        return action
        
    def calculate_action_change_at_site(self, i, j):
        action_change = 0.0

        horizontal_link_1 = self.field.recall_horizontal_link(i, j)
        horizontal_link_2 = self.field.recall_horizontal_link(i, (j + 1) % self.lattice_size)
        vertical_link_1 = self.field.recall_vertical_link(i, j)
        vertical_link_2 = self.field.recall_vertical_link((i + 1) % self.lattice_size, j)

        plaquette_1 = horizontal_link_1 @ horizontal_link_2 @ np.conj(vertical_link_2) @ np.conj(vertical_link_1)

        horizontal_link_3 = self.field.recall_horizontal_link(i, j)
        horizontal_link_4 = self.field.recall_horizontal_link((i + 1) % self.lattice_size, j)
        vertical_link_3 = self.field.recall_vertical_link(i, j)
        vertical_link_4 = self.field.recall_vertical_link(i, (j + 1) % self.lattice_size)

        plaquette_2 = horizontal_link_3 @ np.conj(horizontal_link_4) @ vertical_link_4 @ np.conj(vertical_link_3)

        action_change += 1 - np.real(np.trace(plaquette_1))
        action_change += 1 - np.real(np.trace(plaquette_2))

        return action_change

    def perform_local_update(self, i, j, beta):
        proposed_horizontal_link = random_su2_matrix()
        proposed_vertical_link = random_su2_matrix()

        delta_action = (
            self.calculate_action_change_at_site(i, j) +
            self.calculate_action_change_at_site(i, (j - 1) % self.lattice_size) +
            self.calculate_action_change_at_site((i - 1) % self.lattice_size, j) -
            self.calculate_action_change_at_site(i, j) -
            self.calculate_action_change_at_site(i, (j + 1) % self.lattice_size) -
            self.calculate_action_change_at_site((i + 1) % self.lattice_size, j)
        )

        if delta_action <= 0.0 or np.random.rand() < np.exp(-beta * delta_action):
            self.field.set_horizontal_link(i, j, proposed_horizontal_link)
            self.field.set_vertical_link(i, j, proposed_vertical_link)

        
    def randomize(self):
        for i in range(self.lattice_size):
            for j in range (self.lattice_size):
                
                # Assignes SU(2) matrices to each link in the lattice. One per link.
                horizontal_link = random_su2_matrix()
                vertical_link = random_su2_matrix()
                
                # The self.field represents the Gluon field which will contain the links of the lattice.
                # The set_link will recall a su2 matrix at the link position (i, j). Thus creating a link.
                self.field.set_horizontal_link(i, j, horizontal_link)
                self.field.set_vertical_link(i, j, vertical_link)
                # The above might not work, as it doesn't allow for complex number elements
                
    
    def calculate_wilson_loop(self, x, y, size):
        #start the loop with the identity
        """Calculates the Wilson loop at a given position (x, y) and size.
        The Wilson loop is a closed loop formed by the product of links around a plaquette.
        It provides information about the behavior of quarks and confinement in the gauge theory."""
        loopy_loop = np.eye(2, dtype = np.complex128)
        for i in range(size):
            #Top bit
            horizontal_link = self.field.recall_horizontal_link((x + i) % self.lattice_size, y)
            
            """
            The loop starts from the (x, y) position and moves to the right (x + i, y).
            The link matrix at (x + i, y) is directly multiplied with the loop value,
            maintaining the correct order."""
            loopy_loop = loopy_loop @ horizontal_link
            #Right one
            vertical_link = self.field.recall_vertical_link(x + size, (y + i) % self.lattice_size)
            loopy_loop = loopy_loop @ vertical_link.conj().T
            
            """
            The loop continues from the last position (x + size - 1, y) and 
            moves downward along the right edge (x + size - 1, y + i). Here, we want to 
            multiply the link matrix at (x + size - 1, y + i) with the loop value. However, 
            since the link matrices are SU(2) matrices, their conjugate transpose operation 
            is required to maintain gauge invariance."""
            horizontal_link = self.field.recall_horizontal_link((x + size - i) % self.lattice_size, y + size)
            loopy_loop = loopy_loop @  horizontal_link.conj().T
            #Bottom edge
            vertical_link = self.field.recall_vertical_link(x, (y + size - i) % self.lattice_size)
            loopy_loop = loopy_loop @ vertical_link
            
        return loopy_loop



# Perfom the specified number of updates
lattice_size = 10
num_updates = 1000
beta = 1.2

with open("action.txt", "w") as file:
    pass

with open("wilson_loop.txt", "w") as file:
    pass

with open("su2_matrices.txt", "w") as file:
    pass

lattice = Lattice(lattice_size)
lattice.randomize()

action_history = []
wilson_loop_history = []
wilson_loop_expectation_values = []

wilson = 0

for _ in tqdm(range(num_updates), desc="Metropolis Updates", ncols=100):

    for i in range(lattice_size):
        for j in range(lattice_size):
            lattice.perform_local_update(i, j, beta)
            
    action = lattice.calculate_action()
    action_history.append(action)        
    
    wilson_loop = lattice.calculate_wilson_loop(0, 0, 3)
    wilson_loop_history.append(wilson_loop)
    
    wilson_loop_expectation = np.trace(wilson_loop).real 
    wilson += wilson_loop_expectation + 0.5
    wilson_loop_expectation_values.append(wilson / (_ + 1))

final_action = lattice.calculate_action()

print("Final Action:", final_action)
plt.figure(facecolor='#ECEEF0')
plt.plot(range(num_updates), action_history, color = '#007C7F')
ax = plt.axes()
ax.set_facecolor("#ECEEF0")
plt.xlabel(' Number of Updates ')
plt.ylabel(' Action ')
plt.title(' Action Convergence ')
plt.grid(True)
plt.show()

plt.figure(facecolor='#ECEEF0')
plt.plot(range(num_updates), wilson_loop_expectation_values, 'r', color = '#007C7F')
ax = plt.axes()
ax.set_facecolor("#ECEEF0")
plt.xlabel(' Number of Updates ')
plt.ylabel(' Values of Wilson Loop ')
plt.grid(True)
plt.show()