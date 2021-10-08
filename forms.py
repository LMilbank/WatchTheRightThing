from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo



class LoginForm(FlaskForm):
    username = StringField('username', validators=[DataRequired()])
    listpath = SelectField('list')
    submit = SubmitField('Generate')
