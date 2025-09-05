const express = require('express');
const authController = require('../controllers/authController');
const { authMiddleware } = require('../middleware/auth');
const validationMiddleware = require('../middleware/validation');

const router = express.Router();

// Public routes
router.post('/register', validationMiddleware.validateRegistration, authController.register);
router.post('/login', validationMiddleware.validateLogin, authController.login);
router.post('/refresh', authController.refresh);

// Protected routes
router.get('/profile', authMiddleware, authController.profile);
router.put('/profile', authMiddleware, validationMiddleware.validateProfileUpdate, authController.updateProfile);
router.post('/change-password', authMiddleware, validationMiddleware.validatePasswordChange, authController.changePassword);
router.post('/logout', authMiddleware, authController.logout);

module.exports = router;