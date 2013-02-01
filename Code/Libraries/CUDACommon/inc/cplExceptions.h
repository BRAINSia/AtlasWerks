/* ================================================================
 *
 * AtlasWerks Project
 *
 * Copyright (c) Sarang C. Joshi, Bradley C. Davis, J. Samuel Preston,
 * Linh K. Ha. All rights reserved.  See Copyright.txt or for details.
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notice for more information.
 *
 * ================================================================ */

#include <string>
#include <stdexcept>

namespace cplException {
    class thread_exception :  public std::exception
    {
    protected:
        thread_exception()                : m_err_code(0)  {}
        thread_exception(int err_code): m_err_code(err_code)    {}
    public:
        ~thread_exception() throw() {}
        int native_error() const {
            return m_err_code;
        }
    private:
        int m_err_code;
    };

    class lock_error:public thread_exception
    {
    public:
        lock_error() {}
        lock_error(int err_code):thread_exception(err_code){}
        ~lock_error() throw() {}
        virtual const char* what() const throw() {
            return "cplException::lock_error";
        }
    };

    class thread_resource_error:public thread_exception
    {
    public:
        thread_resource_error() {}
        thread_resource_error(int err_code):thread_exception(err_code){}
        ~thread_resource_error() throw(){}
        virtual const char* what() const throw()    {
            return "cplException::thread_resource_error";
        }
    };

    class unsupported_thread_option: public thread_exception
    {
    public:
        unsupported_thread_option()          {}
        unsupported_thread_option(int err_code): thread_exception(err_code)        {}
        ~unsupported_thread_option() throw() {}
        virtual const char* what() const throw() {
                return "cplException::unsupported_thread_option";
        }
    };
    class invalid_thread_argument:public thread_exception
    {
    public:
        invalid_thread_argument() {}
        invalid_thread_argument(int err_code):thread_exception(err_code) {}
        ~invalid_thread_argument() throw(){}
        virtual const char* what() const throw() {
            return "cplException::invalid_thread_argument";
        }
    };
    class thread_permission_error:public thread_exception
    {
    public:
        thread_permission_error(){}
        thread_permission_error(int err_code):thread_exception(err_code){}
        ~thread_permission_error() throw(){}
        virtual const char* what() const throw(){
            return "cplException::thread_permission_error";
        }
    };
}
