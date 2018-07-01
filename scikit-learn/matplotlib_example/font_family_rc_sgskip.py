import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

fig, ax = plt.subplots()
ax.plot([1, 2.5, 3, 4.5], label='test')
ax.legend()
#plt.savefig('font_family_rc.jpg')
plt.show()
