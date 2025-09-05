const Joi = require('joi');
const logger = require('../utils/logger');

// Validation schemas
const schemas = {
  registration: Joi.object({
    email: Joi.string().email().required().messages({
      'string.email': 'Please provide a valid email address',
      'any.required': 'Email is required'
    }),
    password: Joi.string().min(6).max(100).required().messages({
      'string.min': 'Password must be at least 6 characters long',
      'any.required': 'Password is required'
    }),
    firstName: Joi.string().min(2).max(50).required().messages({
      'string.min': 'First name must be at least 2 characters long',
      'any.required': 'First name is required'
    }),
    lastName: Joi.string().min(2).max(50).required().messages({
      'string.min': 'Last name must be at least 2 characters long',
      'any.required': 'Last name is required'
    }),
    phoneNumber: Joi.string().pattern(/^(\+?6?01)[0-46-9]-*[0-9]{7,8}$/).allow('').messages({
      'string.pattern.base': 'Please provide a valid Malaysian phone number'
    })
  }),

  login: Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().required()
  }),

  profileUpdate: Joi.object({
    firstName: Joi.string().min(2).max(50).required().messages({
      'string.min': 'First name must be at least 2 characters long',
      'any.required': 'First name is required'
    }),
    lastName: Joi.string().min(2).max(50).required().messages({
      'string.min': 'Last name must be at least 2 characters long',
      'any.required': 'Last name is required'
    }),
    phoneNumber: Joi.string().pattern(/^(\+?6?01)[0-46-9]-*[0-9]{7,8}$/).allow('').messages({
      'string.pattern.base': 'Please provide a valid Malaysian phone number'
    })
  }),

  passwordChange: Joi.object({
    currentPassword: Joi.string().required().messages({
      'any.required': 'Current password is required'
    }),
    newPassword: Joi.string().min(6).max(100).required().messages({
      'string.min': 'New password must be at least 6 characters long',
      'any.required': 'New password is required'
    }),
    confirmPassword: Joi.string().valid(Joi.ref('newPassword')).required().messages({
      'any.only': 'Password confirmation does not match',
      'any.required': 'Password confirmation is required'
    })
  }),

  forgotPassword: Joi.object({
    email: Joi.string().email().required().messages({
      'string.email': 'Please provide a valid email address',
      'any.required': 'Email is required'
    })
  }),

  resetPassword: Joi.object({
    token: Joi.string().required().messages({
      'any.required': 'Reset token is required'
    }),
    password: Joi.string().min(6).max(100).required().messages({
      'string.min': 'Password must be at least 6 characters long',
      'any.required': 'Password is required'
    }),
    confirmPassword: Joi.string().valid(Joi.ref('password')).required().messages({
      'any.only': 'Password confirmation does not match',
      'any.required': 'Password confirmation is required'
    })
  })
};

// Validation middleware function
const validate = (schema) => {
  return (req, res, next) => {
    const { error, value } = schema.validate(req.body, {
      abortEarly: false, // Return all validation errors
      stripUnknown: true // Remove unknown fields
    });

    if (error) {
      const errors = error.details.map(detail => ({
        field: detail.path.join('.'),
        message: detail.message
      }));

      logger.warn('Validation error:', {
        path: req.path,
        method: req.method,
        errors: errors,
        body: req.body
      });

      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: errors
      });
    }
    req.body = value;
    next();
  };
};

// Export schemas and validation middleware
module.exports = {
  schemas,
  validate
};